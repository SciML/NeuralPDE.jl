include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))
using ModelingToolkitBase: getdefault
using SymbolicIndexingInterface: getu, getp
using Zygote, LinearAlgebra, Statistics

# `Dense(1, 1)` with weight 1 and bias 0 makes the network the identity, so every
# integral below has a closed-form value to compare against.
const LIN1 = [1.0, 0.0]

function integral_quadrature(prob, block)
    qf, ξs, ws = block.extra_params
    return getdefault(qf), getdefault(ξs), getdefault(ws)
end

@testset "integral terms lower to a registered quadrature" begin
    @parameters t tau
    @variables x(..)
    I = Integral(tau in DomainSets.ClosedInterval(0.0, t))
    chain = Chain(Dense(1, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.2); init_params = LIN1)
    @named sys = PDESystem(
        [I((t - tau) * x(tau)) ~ t^3 / 6], [x(0.0) ~ 0.0],
        [t ∈ Interval(0.0, 1.0)], [t], [x(t)]
    )
    rep = symbolic_discretize(sys, disc)
    md = pinn_metadata(rep)
    block = md.blocks[1]
    res = Symbolics.unwrap(block.residual)
    function has_quadrature(s)
        s = Symbolics.unwrap(s)
        iscall(s) || return false
        operation(s) === NeuralPDE.quadrature && return true
        return any(has_quadrature, arguments(s))
    end
    @test has_quadrature(res)
    qf, ξs, ws = block.extra_params
    @test getdefault(qf) isa NeuralPDE.QuadratureIntegrand
    @test length(block.extra_params) == 3
    pnames = Symbol.(Symbolics.tosymbol.(ModelingToolkit.parameters(rep)))
    @test Symbol(:qf_pde1_1) in pnames
    @test Symbol(:xi_pde1_1) in pnames
    @test Symbol(:wi_pde1_1) in pnames
end

@testset "integral bound variables" begin
    @parameters t tau sigma
    @variables x(..)
    I = Integral(tau in DomainSets.ClosedInterval(0.0, t))
    J = Integral(tau in DomainSets.ClosedInterval(t / 2, t))
    F = Integral(tau in DomainSets.ClosedInterval(0.0, 1.0))
    Dtau = Differential(tau)
    chain = Chain(Dense(1, 1))
    cases = [
        ("convolution", I((t - tau) * x(tau)) ~ t^3 / 6, 1.0e-8),
        ("distinct calls", I(x(tau) + x(t)) ~ 3t^2 / 2, 1.0e-8),
        ("shifted argument", I((t - tau) * x(t - tau)) ~ t^3 / 3, 1.0e-8),
        ("fixed limits", F(x(tau)) ~ 1 / 2, 1.0e-8),
        ("variable limits", J(x(tau)) ~ 3t^2 / 8, 1.0e-8),
        ("derivative", I((t - tau) * Dtau(x(tau))) ~ t^2 / 2, 1.0e-6),
        ("no dependent variable", I(t - tau) ~ t^2 / 2, 1.0e-8),
    ]
    for (name, eq, atol) in cases
        @testset "$name" begin
            disc = PhysicsInformedNN(chain, GridTraining(0.2); init_params = LIN1)
            @named sys = PDESystem(
                [eq], [x(0.0) ~ 0.0], [t ∈ Interval(0.0, 1.0)], [t], [x(t)]
            )
            prob = discretize(sys, disc)
            block = pinn_metadata(prob).blocks[1]
            residual = getu(prob, block.residual)(prob)
            @test residual ≈ zeros(size(residual)) atol = atol
        end
    end
end

@testset "nested integrals" begin
    @parameters t tau sigma
    @variables x(..)
    chain = Chain(Dense(1, 1))
    cases = [
        # ∫_0^t ∫_0^τ x(σ) dσ dτ = t³ / 6
        ("nested", Integral(tau in DomainSets.ClosedInterval(0.0, t))(
            Integral(sigma in DomainSets.ClosedInterval(0.0, tau))(x(sigma)))),
        # ∫_0^t ∫_0^t x(σ) dσ dt = t³ / 6; the integrating variable shadows the outer t
        ("nested shadowed bound", Integral(t in DomainSets.ClosedInterval(0.0, t))(
            Integral(sigma in DomainSets.ClosedInterval(0.0, t))(x(sigma)))),
    ]
    for (name, term) in cases
        @testset "$name" begin
            disc = PhysicsInformedNN(chain, GridTraining(0.2); init_params = LIN1)
            @named sys = PDESystem(
                [term ~ t^3 / 6], [x(0.0) ~ 0.0],
                [t ∈ Interval(0.0, 1.0)], [t], [x(t)]
            )
            prob = discretize(sys, disc)
            block = pinn_metadata(prob).blocks[1]
            residual = getu(prob, block.residual)(prob)
            @test residual ≈ zeros(size(residual)) atol = 1.0e-8
        end
    end
end

@testset "multidimensional integrating variables" begin
    @parameters t tau sigma
    @variables x(..)
    I = Integral((tau, sigma) in DomainSets.ProductDomain(
        DomainSets.ClosedInterval(0.0, t), DomainSets.ClosedInterval(0.0, t)))
    chain = Chain(Dense(1, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.2); init_params = LIN1)
    @named sys = PDESystem(
        [I((t - tau) * (t - sigma) * x(tau)) ~ t^5 / 12], [x(0.0) ~ 0.0],
        [t ∈ Interval(0.0, 1.0)], [t], [x(t)]
    )
    prob = discretize(sys, disc)
    block = pinn_metadata(prob).blocks[1]
    @test size(getdefault(block.extra_params[2]), 1) == 2
    residual = getu(prob, block.residual)(prob)
    @test residual ≈ zeros(size(residual)) atol = 1.0e-8
end

@testset "infinite integration bounds" begin
    @parameters t tau
    @variables x(..)
    chain = Chain(Dense(1, 1))
    cases = [
        # ∫_t^∞ τ e^{-τ} dτ = (t + 1) e^{-t}
        ("semi-infinite upper", Integral(tau in DomainSets.ClosedInterval(t, Inf)),
         x(tau) * exp(-tau), X -> (X .+ 1) .* exp.(-X)),
        # ∫_{-∞}^t τ e^{-τ²} dτ = -e^{-t²} / 2
        ("semi-infinite lower", Integral(tau in DomainSets.ClosedInterval(-Inf, t)),
         x(tau) * exp(-tau^2), X -> -exp.(-X .^ 2) ./ 2),
        # ∫_{-∞}^{∞} τ² e^{-τ²} dτ = √π / 2
        ("infinite", Integral(tau in DomainSets.ClosedInterval(-Inf, Inf)),
         x(tau)^2 * exp(-tau^2), X -> fill(sqrt(pi) / 2, 1, size(X, 2))),
    ]
    for (name, I, integrand, expected) in cases
        @testset "$name" begin
            disc = PhysicsInformedNN(chain, GridTraining(0.2); init_params = LIN1)
            @named sys = PDESystem(
                [I(integrand) ~ 0],
                [x(0.0) ~ 0.0], [t ∈ Interval(0.0, 1.0)], [t], [x(t)]
            )
            prob = discretize(sys, disc)
            block = pinn_metadata(prob).blocks[1]
            F, ξ, w = integral_quadrature(prob, block)
            xs = block.xs
            X = xs === nothing ? zeros(0, 1) : getp(prob, xs)(prob)
            out = NeuralPDE.quadrature(F, X, prob.u0, ξ, w)
            @test out ≈ expected(X) atol = 2.0e-4
        end
    end
end

@testset "integral of networks with different signatures" begin
    @parameters t s tau
    @variables x(..) y(..)
    I = Integral(tau in DomainSets.ClosedInterval(0.0, t))
    chains = [Chain(Dense(1, 1)), Chain(Dense(2, 1))]
    disc = PhysicsInformedNN(
        chains, GridTraining(0.2); init_params = [[1.0, 0.0], [1.0, 1.0, 0.0]])
    @named sys = PDESystem(
        [y(0.0, t) + I(x(tau) + y(s, tau)) ~ 0], [x(0.0) ~ 0.0, y(0.0, s) ~ s],
        [t ∈ Interval(0.0, 1.0), s ∈ Interval(0.0, 1.0)], [t, s], [x(t), y(t, s)]
    )
    prob = discretize(sys, disc)
    block = pinn_metadata(prob).blocks[1]
    X = getp(prob, block.xs)(prob)
    residual = getu(prob, block.residual)(prob)
    @test residual ≈ X[1:1, :] .* (X[2:2, :] .+ X[1:1, :] .+ 1) atol = 1.0e-8
end

@testset "gradients flow through quadrature terms" begin
    @parameters t tau
    @variables x(..)
    I = Integral(tau in DomainSets.ClosedInterval(0.0, t))
    chain = Chain(Dense(1, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.2); init_params = LIN1)
    @named sys = PDESystem(
        [I((t - tau) * x(tau)) ~ t^3 / 6], [x(0.0) ~ 0.0],
        [t ∈ Interval(0.0, 1.0)], [t], [x(t)]
    )
    prob = discretize(sys, disc)
    block = pinn_metadata(prob).blocks[1]
    F, ξ, w = integral_quadrature(prob, block)
    X = getp(prob, block.xs)(prob)
    g = only(Zygote.gradient(θ -> sum(NeuralPDE.quadrature(F, X, θ, ξ, w)), prob.u0))
    # x(τ) = w τ + b, so d/dw of ∫ (t - τ) x(τ) dτ = t^3 / 6 and d/db = t^2 / 2
    @test g[1] ≈ sum(X .^ 3) / 6
    @test g[2] ≈ sum(X .^ 2) / 2
    @test Zygote.gradient(θ -> prob.f(θ, prob.p), prob.u0)[1] isa AbstractVector
end

@testset "integral_alg validation" begin
    chain = Chain(Dense(1, 1))
    @test_throws ArgumentError PhysicsInformedNN(
        chain, GridTraining(0.1); integral_alg = Integrals.QuadGKJL())
    disc = PhysicsInformedNN(
        chain, QuadratureTraining(; quadrature_alg = GaussLegendre(n = 8)))
    @test NeuralPDE._integral_alg(disc) isa GaussLegendre
end

@testset "integro-differential equation (BFGS)" begin
    @parameters t tau
    @variables i(..)
    Di = Differential(t)
    Ii = Integral(tau in DomainSets.ClosedInterval(0.0, t))
    eq = Di(i(t)) + 2 * i(t) + 5 * Ii(i(tau)) ~ 1
    bcs = [i(0.0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]
    chain = Chain(Dense(1, 15, σ), Dense(15, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.1); rng = Xoshiro(110))
    @named pde_system = PDESystem(eq, bcs, domains, [t], [i(t)])
    prob = discretize(pde_system, disc)
    sol = train(prob; adam_iters = 100, bfgs_iters = 100)
    analytic(v) = exp(-v) * sin(2v) / 2
    ts = 0.01:0.05:2.0
    @test mean(abs2, [sol(v; dv = i(t)) - analytic(v) for v in ts]) < 0.02
end
