using NeuralPDE, SciMLBase
using Test

# Must stay at top level: `Reactant.@compile` is a macro, so `Reactant` has to be
# resolvable at macroexpansion time. The @testset below is a single top-level
# expression and is macroexpanded before any of its body runs, so moving this import
# inside it (as the other test files do with their imports) would be too late.
using Reactant

# Correctness of the Reactant/Enzyme path in NNODE, in three layers:
#
#   1. `reactant_inner_loss` traces and compiles, and its batched RHS agrees with the
#      legacy `inner_loss`;
#   2. the all-ones forward-mode seed in `enzyme_ode_dfdx` recovers the diagonal dφ/dt;
#   3. the outer Enzyme reverse gradient compiled inside `build_reactant_grid_objective`
#      matches central finite differences of that same compiled loss.
#
# The ordering is deliberate and self-localizing: 1 failing means the traced loop itself is
# wrong; 1 passing while 2 or 3 fails means the loop is fine and it is the differentiation.
#
# Everything compiled: plain Enzyme outside Reactant is a separate code path with its own
# history of trouble, so asserting on it here would let an unrelated failure turn CI red
# while the production path is correct. Run it as a local probe if a test below fails and
# you need to know whether the seed idea or the Reactant integration is at fault.
#
# Everything is Float64: the references are finite-precision comparisons and Float32
# noise is the same order as the quantities under test.
@testset "Reactant NNODE" begin
    using Random, Lux, ComponentArrays, ForwardDiff, LinearAlgebra
    using Optimisers: Adam

    Random.seed!(100)

    function setup_phi(u0, n_out)
        chain = Chain(Dense(1, 5, tanh), Dense(5, n_out))
        ps, _ = Lux.setup(Random.default_rng(), chain)
        ps_ca = ComponentArray(ps)
        ps64 = ComponentArray(Float64.(getdata(ps_ca)), getaxes(ps_ca))
        phi, _ = NeuralPDE.generate_phi_θ(chain, 0.0, u0, ps64)
        return phi, ComponentArray(; depvar = ps64)
    end

    # Independent reference for dφ/dt at each grid point.
    #
    # The legacy batched path cannot serve as the reference here: for a vector `t`,
    # `ode_dfdx(..., autodiff = true)` returns `ForwardDiff.jacobian`, which is N×N for
    # scalar u0 (silently broadcasting against the 1×N residual) and (dim*N)×N otherwise
    # (a dimension mismatch). That is what `generate_loss(::GridTraining, ...)` guards
    # against with its ArgumentError. So the diagonal is built one point at a time.
    #
    # `ForwardDiff.derivative` handles both scalar- and array-valued φ, so this is
    # uniform across the two u0 cases: scalar u0 gives 1×N, vector u0 gives dim×N. That
    # is exactly the shape the all-ones JVP must reproduce.
    function reference_dphi_dt(phi, θ, ts)
        return reduce(hcat, [ForwardDiff.derivative(τ -> phi(τ, θ), tᵢ) for tᵢ in ts])
    end

    function reference_residual(phi, f, p, θ, ts, dphi_dt)
        out = phi(ts, θ)
        fs = if phi.u0 isa Number
            reduce(hcat, [f(out[i], p, tᵢ) for (i, tᵢ) in enumerate(ts)])
        else
            reduce(hcat, [f(out[:, i], p, tᵢ) for (i, tᵢ) in enumerate(ts)])
        end
        return sum(abs2, fs .- dphi_dt) / length(ts)
    end

    cases = (
        (name = "scalar u0", u0 = 0.5, n_out = 1, f = (u, p, t) -> cospi(2t)),
        (
            name = "vector u0", u0 = [0.5, -0.2], n_out = 2,
            f = (u, p, t) -> [cospi(2t), -u[1]],
        ),
    )

    @testset "$(case.name)" for case in cases
        (; u0, n_out, f) = case
        p = nothing

        phi, θ = setup_phi(u0, n_out)
        ts = collect(0.0:0.25:1.0)
        ref = reference_dphi_dt(phi, θ, ts)

        ts_dev = Reactant.to_rarray(ts)
        tangent_dev = Reactant.to_rarray(fill(one(eltype(ts)), size(ts)))
        θ_dev = Reactant.to_rarray(θ)

        # 1. Smoke test. No Enzyme involved: does φ survive tracing, and does the batched
        #    RHS assembly in `reactant_inner_loss` agree with the legacy `inner_loss`?
        #    Both take the finite-difference branch, so they compute an identical formula
        #    and should differ only by XLA floating-point reassociation.
        #
        #    This is the test most likely to fail first, via
        #    phi(t, θ) -> safe_get_device(θ) -> safe_expand(dev, t): if MLDataDevices
        #    does not recognize traced parameters, `safe_get_device` falls back to
        #    CPUDevice and `safe_expand` attempts to materialize a traced array mid-trace.
        @testset "compiles and matches legacy residual (autodiff = false)" begin
            expected = NeuralPDE.inner_loss(phi, f, false, ts, θ, p, false)

            function fd_loss(q, tt, tangent)
                return NeuralPDE.reactant_inner_loss(phi, f, false, tt, tangent, q, p, false)
            end

            compiled = Reactant.@compile fd_loss(θ_dev, ts_dev, tangent_dev)
            got = Reactant.to_number(compiled(θ_dev, ts_dev, tangent_dev))

            @test got ≈ expected rtol = 1.0e-8 atol = 1.0e-10
        end

        # 2. φ is elementwise in t, so its Jacobian w.r.t. t is diagonal and a single
        #    forward pass seeded with ones recovers every du_i/dt_i. If this fails, the
        #    seed in `build_reactant_grid_objective` is wrong and nothing downstream of
        #    it means anything.
        @testset "tangent seed vs per-point ForwardDiff" begin
            dfdx(q, tt, tangent) = NeuralPDE.enzyme_ode_dfdx(phi, tt, q, tangent)

            compiled = Reactant.@compile dfdx(θ_dev, ts_dev, tangent_dev)
            got = Array(compiled(θ_dev, ts_dev, tangent_dev))

            @test size(got) == size(ref)
            @test got ≈ ref rtol = 1.0e-8 atol = 1.0e-10
        end

        # 3. The full residual on the autodiff = true path, against a reference built
        #    from the per-point ForwardDiff diagonal. Separates "is dφ/dt right" (2) from
        #    "is the residual assembled correctly around it" (here).
        @testset "residual matches reference (autodiff = true)" begin
            expected = reference_residual(phi, f, p, θ, ts, ref)

            function ad_loss(q, tt, tangent)
                return NeuralPDE.reactant_inner_loss(phi, f, true, tt, tangent, q, p, false)
            end

            compiled = Reactant.@compile ad_loss(θ_dev, ts_dev, tangent_dev)
            got = Reactant.to_number(compiled(θ_dev, ts_dev, tangent_dev))

            @test got ≈ expected rtol = 1.0e-8 atol = 1.0e-10
        end
    end

    # ---------------------------------------------------------------------------------
    # Layer 3: the outer reverse gradient, i.e. `Enzyme.gradient(Reverse, scalar_loss, …)`
    # as compiled inside `build_reactant_grid_objective`. Nothing above reaches it -- the
    # layers there stop at the forward JVP and the residual -- and a training run only
    # shows that the loss decreases, which a subtly wrong gradient can also do.
    #
    # Both sides come from the production objects, so the finite differences and the
    # gradient differentiate the same compiled loss by construction:
    #
    #   L(θ) = obj.objective(θ)     -> the compiled `scalar_loss`
    #   g    = obj.gradient!(G, θ)  -> Enzyme.gradient(Reverse, scalar_loss, ...)
    #
    # Complex parameters are checked along a real and an imaginary direction separately:
    # for a real-valued L of complex θ these probe different halves of the Wirtinger pair,
    # so a convention mismatch would surface in one and not the other. Both matching is
    # what pins Enzyme's convention here to ∇ₓL + i∇ᵧL -- the same one Zygote uses -- and
    # makes `real(dot(g, v))` the directional derivative with no correction factor.
    #
    # The two h values bracket the Float64 central-difference optimum (~6e-6), where a
    # development sweep over 1e-4 … 1e-6 measured relative errors of 6e-11 or better in all
    # three cases. `grad_rtol` is four orders looser so platform floating-point variation
    # cannot make this flaky, while any realistic breakage -- a dropped term, a sign flip,
    # a factor of two from the wrong Wirtinger convention -- is off by O(1) and still fails.
    # Shared by layers 3 and 4, which both drive the production
    # `build_reactant_grid_objective` rather than the pieces underneath it.
    obj_tspan = (0.0, 1.0)
    obj_strategy = GridTraining(0.25)

    function setup_theta(chain, u0, T)
        Random.seed!(100)
        ps, _ = Lux.setup(Random.default_rng(), chain)
        ps_ca = ComponentArray(ps)
        ps_T = ComponentArray(T.(getdata(ps_ca)), getaxes(ps_ca))
        phi, init_params = NeuralPDE.generate_phi_θ(chain, obj_tspan[1], u0, ps_T)
        return phi, ComponentArray(; depvar = init_params)
    end

    build_objective(phi, f, θ) = NeuralPDE.build_reactant_grid_objective(
        phi, f, true, obj_tspan, obj_strategy, nothing, false, θ
    )

    real_case = (
        T = Float64, u0 = 0.0,
        f = (u, p, t) -> cospi(2t) - u,
        chain = Chain(Dense(1, 4, tanh), Dense(4, 1)),
    )
    complex_case = (
        T = ComplexF64, u0 = ComplexF64(1.0, 0.0),
        f = (u, p, t) -> im * u,
        chain = Chain(
            Dense(1, 4, tanh; init_weight = kaiming_normal(ComplexF64)),
            Dense(4, 1; init_weight = kaiming_normal(ComplexF64)),
        ),
    )

    @testset "outer reverse gradient" begin
        hs = (1.0e-5, 3.0e-6)
        grad_rtol = 1.0e-6

        """Deterministic normalized direction shaped like θ."""
        function direction(θ, T; imaginary = false)
            rng = Xoshiro(1234)
            raw = randn(rng, Float64, length(θ))
            v = ComponentArray(imaginary ? (im .* raw) : T.(raw), getaxes(θ))
            return v ./ sqrt(sum(abs2, v))
        end

        function objective_and_gradient(phi, f, θ)
            obj = build_objective(phi, f, θ)
            G = similar(θ)
            fill!(G, zero(eltype(G)))
            obj.gradient!(G, θ)
            return obj, G
        end

        function check_direction(obj, θ, v, ad)
            for h in hs
                fd = (obj.objective(θ .+ h .* v) - obj.objective(θ .- h .* v)) / (2h)
                @test ad ≈ fd rtol = grad_rtol
            end
        end

        @testset "real Float64 parameters" begin
            (; T, u0, f, chain) = real_case
            phi, θ = setup_theta(chain, u0, T)
            obj, g = objective_and_gradient(phi, f, θ)

            @test obj.objective(θ) isa Real
            @test eltype(g) == Float64
            @test all(isfinite, g)

            v = direction(θ, Float64)
            check_direction(obj, θ, v, dot(g, v))
        end

        @testset "complex ComplexF64 parameters" begin
            (; T, u0, f, chain) = complex_case
            phi, θ = setup_theta(chain, u0, T)
            obj, g = objective_and_gradient(phi, f, θ)

            # A complex network still has a real loss -- it is a sum of `abs2`. If this
            # ever returns Complex, every consumer that orders the objective breaks:
            # NNODE's own `l < abstol` callback and `save_best` inside
            # OptimizationOptimisers, the latter third-party and unpatchable from here.
            # The gradient, by contrast, must stay complex with a nonzero imaginary part.
            @test obj.objective(θ) isa Real
            @test eltype(g) <: Complex
            @test all(isfinite, g)
            @test any(!iszero, imag.(g))

            @testset "real direction" begin
                v = direction(θ, ComplexF64)
                check_direction(obj, θ, v, real(dot(g, v)))
            end

            @testset "imaginary direction" begin
                v = direction(θ, ComplexF64; imaginary = true)
                check_direction(obj, θ, v, real(dot(g, v)))
            end
        end
    end

    # ---------------------------------------------------------------------------------
    # Layer 4: parameter staging. `objective`/`gradient!` copy into a `θ_dev` built once
    # when the objective is constructed, rather than allocating a fresh device array per
    # call. That is the kind of optimization that works almost always, and an end-to-end
    # solve would not reliably catch a regression: a compiled thunk pinned to the
    # tracing-time parameters still produces a moving loss, just the wrong one.
    #
    # The behavioural assertions are the load-bearing ones -- the objective must observe
    # each new value, a gradient call must not corrupt the buffer for the next objective,
    # and the gradient must be both reproducible and responsive. Exact equality is used
    # deliberately: the same input through the same compiled thunk should be bit-identical,
    # so any drift is itself the finding.
    @testset "parameter buffer reuse" begin
        @testset "$(name)" for (name, case) in (
                ("real Float64", real_case), ("complex ComplexF64", complex_case),
            )
            (; T, u0, f, chain) = case
            phi, θ1 = setup_theta(chain, u0, T)
            θ2 = θ1 .* 1.1

            # The buffer must be reused, not replaced: `copyto!` has to write through to
            # the array the ComponentArray already wraps.
            rθ = Reactant.to_rarray(θ1)
            buf = NeuralPDE._reactant_data(rθ)
            axes_before = getaxes(rθ)
            copyto!(NeuralPDE._reactant_data(rθ), NeuralPDE._reactant_data(θ2))

            @test NeuralPDE._reactant_data(rθ) === buf
            @test Array(buf) ≈ getdata(θ2)
            @test getaxes(rθ) == axes_before
            @test eltype(rθ) == T

            obj = build_objective(phi, f, θ1)
            G = similar(θ1)

            L1 = obj.objective(θ1)
            L2 = obj.objective(θ2)
            @test L1 != L2
            @test obj.objective(θ1) == L1

            fill!(G, zero(eltype(G)))
            obj.gradient!(G, θ1)
            g1 = copy(G)

            @test obj.objective(θ2) == L2
            @test obj.objective(θ1) == L1

            fill!(G, zero(eltype(G)))
            obj.gradient!(G, θ1)
            @test G == g1

            fill!(G, zero(eltype(G)))
            obj.gradient!(G, θ2)
            @test G != g1
        end
    end

    # ---------------------------------------------------------------------------------
    # Layer 5: the whole production path, through `solve` with default settings.
    #
    # Layers 1-4 stop at the objective and gradient. Everything between them and a real
    # training run is untested by those: the OptimizationFunction calling convention, the
    # NNODE callback, `save_best` inside OptimizationOptimisers, and the optimizer's
    # parameter update. That gap has already hidden two real bugs -- an objective-arity
    # mismatch under `SciMLBase.NoAD`, and a complex-typed scalar loss that `<` cannot
    # order -- neither of which any unit-level assertion above would have caught.
    #
    # Complex parameters are used deliberately: one solve then covers a real-valued
    # objective, a complex gradient, the calling convention, the callback, `save_best`,
    # and a genuine parameter update in both components.
    @testset "end-to-end solve" begin
        (; T, u0, f, chain) = complex_case
        _, θ0 = setup_theta(chain, u0, T)

        Random.seed!(100)
        ps, _ = Lux.setup(Random.default_rng(), chain)
        ps_ca = ComponentArray(ps)
        psT = ComponentArray(T.(getdata(ps_ca)), getaxes(ps_ca))

        prob = ODEProblem(f, u0, obj_tspan)
        alg = NNODE(
            chain, Adam(0.01), psT;
            strategy = obj_strategy, autodiff = true, batch = true, reactant = true
        )

        # Default settings -- in particular save_best is left at its default.
        sol = solve(prob, alg; verbose = false, maxiters = 2)

        @test sol.k.objective isa Real
        @test isfinite(sol.k.objective)
        @test eltype(sol.k.u) == T
        @test all(isfinite, sol.k.u)

        # The optimizer must actually have moved, in both components.
        @test !all(sol.k.u .≈ θ0)
        @test maximum(abs, real.(sol.k.u) .- real.(θ0)) > 0
        @test maximum(abs, imag.(sol.k.u) .- imag.(θ0)) > 0

        # And the opt-in must refuse configurations it does not support rather than
        # silently falling back to the legacy path.
        @test_throws ArgumentError solve(
            prob,
            NNODE(
                chain, Adam(0.01), psT;
                strategy = StochasticTraining(10), batch = true, reactant = true
            );
            verbose = false, maxiters = 2
        )
    end
end
