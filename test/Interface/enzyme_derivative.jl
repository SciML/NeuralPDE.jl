using NeuralPDE, Lux, Random, Test, Enzyme, ForwardDiff, Zygote, Symbolics
using ComponentArrays: ComponentArray
using DomainSets: Interval
using ModelingToolkitBase: @named, getdefault
using SymbolicIndexingInterface: getp, getu
using Statistics: mean
using ADTypes: AutoEnzyme

function reference_partial(f, x, θ, ::Tuple{})
    return only(f(reshape(x, :, 1), θ))
end
function reference_partial(f, x, θ, directions::Tuple)
    e = [i == first(directions) for i in eachindex(x)]
    return ForwardDiff.derivative(
        t -> reference_partial(f, x .+ t .* e, θ, Base.tail(directions)), zero(eltype(x))
    )
end

exponential_network(X, θ) = θ[1] .* exp.(θ[2] .* X[1:1, :] .+ θ[3] .* X[2:2, :])

@testset "batched JVP values, code generation, and pullbacks" begin
    X = [0.1 0.2 0.4; 0.3 0.4 0.5]
    θ = [0.7, 0.5, 0.3]
    for ds in ((1,), (1, 1), (1, 1, 1), (1, 1, 1, 1), (1, 2), (1, 2, 1, 2))
        directions = Val(map(d -> (d,), ds))
        exact(p) = exponential_network(X, p) .* prod(p[d + 1] for d in ds)
        actual(p) = NeuralPDE.nn_jvp(exponential_network, X, p, directions, 1)
        @test actual(θ) ≈ exact(θ) rtol = 128eps(Float64)
        objective(p) = sum(abs2, actual(p))
        expected = ForwardDiff.gradient(p -> sum(abs2, exact(p)), θ)
        gradient = zero(θ)
        Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(objective), Enzyme.Active, Enzyme.Duplicated(θ, gradient))
        @test gradient ≈ expected rtol = 512eps(Float64)
        @test only(Zygote.gradient(objective, θ)) ≈ expected rtol = 512eps(Float64)
    end
    @variables xs[1:2, 1:3] ps[1:3]
    expr = NeuralPDE.nn_jvp(exponential_network, xs, ps, Val(((1,), (2,))), 1)
    generated = Symbolics.build_function(expr, xs, ps; expression = Val(false))[1]
    @test generated(X, θ) ≈ exponential_network(X, θ) .* θ[2] .* θ[3]
    @test size(expr) == (1, 3)
    @test NeuralPDE.nn_jvp(exponential_network, X, θ, Val(((1, 2),)), 1) ≈
        exponential_network(X, θ) .* (θ[2] + θ[3])
    input_gradient = only(Zygote.gradient(z -> sum(generated(z, θ)), X))
    @test input_gradient ≈ vcat(
        exponential_network(X, θ) .* θ[2]^2 .* θ[3],
        exponential_network(X, θ) .* θ[2] .* θ[3]^2
    )
    @test only(Zygote.gradient(p -> sum(generated(X, p)), θ)) ≈
        ForwardDiff.gradient(p -> sum(exponential_network(X, p) .* p[2] .* p[3]), θ)
end

@testset "Lux sin activation uses the same primal under both AD paths" begin
    layer = Dense(1, 1, sin)
    ps = (weight = ones(1, 1), bias = zeros(1))
    st = Lux.initialstates(Xoshiro(1), layer)
    X = reshape(collect(range(-1.0, 1.0; length = 17)), 1, :)
    f(z) = first(Lux.apply(layer, z, ps, st))
    dual_primal = ForwardDiff.value.(f(ForwardDiff.Dual.(X, one.(X))))
    _, enzyme_primal = Enzyme.autodiff(
        Enzyme.ForwardWithPrimal, Enzyme.Const(f), Enzyme.Duplicated,
        Enzyme.Duplicated(X, one.(X))
    )
    @test isequal(f(X), sin.(X))
    @test isequal(f(X), dual_primal)
    @test isequal(enzyme_primal, dual_primal)
end

@testset "Enzyme PDE residuals and parameter gradients" begin
    @parameters x y
    @variables u(..)
    Dx, Dy = Differential(x), Differential(y)
    @named poisson = PDESystem(
        Dx(Dx(u(x, y))) + Dy(Dy(u(x, y))) ~ -sinpi(x) * sinpi(y),
        [u(0, y) ~ 0.0, u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0],
        [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)], [x, y], [u(x, y)]
    )
    @named third_order = PDESystem(
        (Dx^3)(u(x)) ~ cospi(x),
        [u(0.0) ~ 0.0, u(1.0) ~ cospi(1.0), Dx(u(1.0)) ~ 1.0],
        [x ∈ Interval(0.0, 1.0)], [x], [u(x)]
    )
    @named mixed = PDESystem(
        Dx(Dy(u(x, y))) + Dy(Dx(u(x, y))) ~ 2,
        [u(x, 0) ~ 0.0, u(0, y) ~ 0.0],
        [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)], [x, y], [u(x, y)]
    )
    @testset "$activation activation" for activation in (sin, tanh)
        for (system, dim, order) in ((poisson, 2, 2), (third_order, 1, 3), (mixed, 2, 2))
            chain = Chain(Dense(dim, 3, activation), Dense(3, 1))
            θ0 = Float64.(collect(ComponentArray(Lux.initialparameters(Xoshiro(42), chain)))) ./ 4
            probs = map((EnzymeForwardDerivative(), FiniteDifferenceDerivative())) do derivative
                disc = PhysicsInformedNN(
                    chain, GridTraining(0.25); derivative, init_params = θ0, rng = Xoshiro(1)
                )
                discretize(system, disc; adtype = AutoEnzyme())
            end
            prob, fdprob = probs
            md = pinn_metadata(prob)
            f = getdefault(only(md.networks).NN)
            points = [b.xs === nothing ? nothing : getp(prob, b.xs)(prob) for b in md.blocks]
            function reference_residuals(p)
                X = points[1]
                interior = if system === poisson
                    [
                        reference_partial(f, z, p, (1, 1)) + reference_partial(f, z, p, (2, 2)) +
                            sinpi(z[1]) * sinpi(z[2]) for z in eachcol(X)
                    ]
                elseif system === third_order
                    [reference_partial(f, z, p, (1, 1, 1)) - cospi(z[1]) for z in eachcol(X)]
                else
                    [
                        reference_partial(f, z, p, (1, 2)) + reference_partial(f, z, p, (2, 1)) - 2
                            for z in eachcol(X)
                    ]
                end
                boundary = if system === poisson
                    [
                        vec(f(vcat(zero(points[2]), points[2]), p)),
                        vec(f(vcat(one.(points[3]), points[3]), p)),
                        vec(f(vcat(points[4], zero(points[4])), p)),
                        vec(f(vcat(points[5], one.(points[5])), p)),
                    ]
                elseif system === third_order
                    [
                        vec(f(zeros(1, 1), p)), vec(f(ones(1, 1), p)) .+ 1,
                        [reference_partial(f, [1.0], p, (1,)) - 1],
                    ]
                else
                    [
                        vec(f(vcat(points[2], zero(points[2])), p)),
                        vec(f(vcat(zero(points[3]), points[3]), p)),
                    ]
                end
                return [interior, boundary...]
            end
            θ = prob.u0
            references = reference_residuals(θ)
            # Central stencils have O(h²) truncation and O(eps/h^order) roundoff.
            # Mixed stencils compose two first-order steps, each eps^(1/3).
            h = system === mixed ? eps(Float64)^(1 / 3) : eps(Float64)^(1 / (2 + order))
            fd_atol = 32 * (h^2 + eps(Float64) / h^order)
            for (i, block) in enumerate(md.blocks)
                actual = vec(getu(prob, block.residual)(prob))
                @test actual ≈ references[i] atol = 4096eps(Float64) rtol = 4096eps(Float64)
                baseline = vec(getu(fdprob, pinn_metadata(fdprob).blocks[i].residual)(fdprob))
                @test actual ≈ baseline atol = fd_atol rtol = fd_atol
            end
            reference_loss(p) = sum(r -> mean(abs2, r), reference_residuals(p))
            @test prob.f(θ, prob.p) ≈ reference_loss(θ) rtol = 4096eps(Float64)
            expected = ForwardDiff.gradient(reference_loss, θ)
            actual = zero(θ)
            objective(p) = prob.f(p, prob.p)
            Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(objective), Enzyme.Active, Enzyme.Duplicated(θ, actual))
            # NNlib's Float64 tanh_fast polynomial and its Dual fallback (tanh) differ.
            # At x=0.1, 256-bit fourth derivatives are 1.5553210429164255 and
            # 1.5553210414847942: ~9.2e-10 relative; 2e-9 allows twice that discrepancy.
            # Polynomial source: https://github.com/FluxML/NNlib.jl/blob/v0.9.45/src/activations.jl
            gradient_rtol = activation === tanh ? 2.0e-9 : 4096eps(Float64)
            @test actual ≈ expected atol = 4096eps(Float64) rtol = gradient_rtol
            @test only(Zygote.gradient(objective, θ)) ≈ expected atol = 4096eps(Float64) rtol = gradient_rtol
            baseline = ForwardDiff.gradient(p -> fdprob.f(p, fdprob.p), θ)
            @test actual ≈ baseline atol = fd_atol rtol = fd_atol
        end
    end
end
