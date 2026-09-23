using DomainSets, Lux, ModelingToolkit, ModelingToolkitBase, NeuralPDE
using Optimization, OptimizationOptimisers, Random, SciMLBase, Test
using SymbolicIndexingInterface: getp
import DomainSets: Interval

function adaptive_test_problem()
    @variables z
    @parameters w1 w2
    sys = complete(OptimizationSystem([(z - 1)^2, 3 * (z + 1)^2], [z], [w1, w2]; name = :adaptive_test))
    prob = OptimizationProblem(
        sys, [z => 0.0, w1 => 1.0, w2 => 1.0]; weights = [w1, w2], grad = true
    )
    return prob, [w1, w2]
end

function adaptive_state(prob, iter, z)
    return (iter, u = [z], p = prob.p)
end

adaptive_weights(prob, weights) = [getp(prob.f.sys, w)(prob.p) for w in weights]

@testset "Adaptive loss callback updates" begin
    @testset "GradientScale" begin
        prob, weights = adaptive_test_problem()
        callback = GradientScaleAdaptiveLoss(prob, weights; inertia = 0.0)
        callback(adaptive_state(prob, 1, 0.0), 0.0)
        @test adaptive_weights(prob, weights) ≈ [3.0, 1.0]
    end

    @testset "MiniMax" begin
        prob, weights = adaptive_test_problem()
        callback = MiniMaxAdaptiveLoss(prob, weights; optimizer = Adam(0.5))
        callback(adaptive_state(prob, 1, 0.0), 0.0)
        @test adaptive_weights(prob, weights) ≈ [1.5, 1.5] atol = 1.0e-6
    end

    @testset "SoftAdapt" begin
        prob, weights = adaptive_test_problem()
        callback = SoftAdaptAdaptiveLoss(prob, weights; α = 0.1)
        callback(adaptive_state(prob, 1, 0.0), 0.0)
        callback(adaptive_state(prob, 2, 0.5), 0.0)
        scores = 0.1 .* [-0.75, 1.25]
        expected = 2 .* exp.(scores .- maximum(scores)) ./ sum(exp.(scores .- maximum(scores)))
        @test adaptive_weights(prob, weights) ≈ expected
    end

    @testset "ReLoBRaLo" begin
        prob, weights = adaptive_test_problem()
        callback = ReLoBRaLoAdaptiveLoss(
            prob, weights; α = 0.5, β = 0.0, temperature = 0.7, rng = Xoshiro(2)
        )
        callback(adaptive_state(prob, 1, 0.0), 0.0)
        callback(adaptive_state(prob, 2, 0.25), 0.0)
        callback(adaptive_state(prob, 3, 0.5), 0.0)
        initial = [1.0, 3.0]
        previous = [0.5625, 4.6875]
        current = [0.25, 6.75]
        scores0 = current ./ (0.7 .* initial)
        scores1 = current ./ (0.7 .* previous)
        initial_balance = 2 .* exp.(scores0 .- maximum(scores0)) ./ sum(exp.(scores0 .- maximum(scores0)))
        previous_balance = 2 .* exp.(scores1 .- maximum(scores1)) ./ sum(exp.(scores1 .- maximum(scores1)))
        expected = 0.5 .* initial_balance .+ 0.5 .* previous_balance
        @test adaptive_weights(prob, weights) ≈ expected
    end
end

function poisson_error(callback_factory; initial_weights = nothing)
    @parameters x y w[1:5]
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)
    bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    @named pdesys = PDESystem(eq, bcs, domains, [x, y], [u(x, y)], collect(w))
    weight_symbols = collect(w)
    chain = Chain(Dense(2, 40, tanh), Dense(40, 40, tanh), Dense(40, 1))
    disc = PhysicsInformedNN(chain, StochasticTraining(256); rng = Xoshiro(60))
    p = collect(weight_symbols .=> (initial_weights === nothing ? ones(5) : initial_weights))
    prob = discretize(pdesys, disc; weights = weight_symbols, p)
    callback = callback_factory(prob, weight_symbols)
    solution = solve(prob, Adam(0.03); maxiters = 2000, callback)
    xs = 0:0.01:1
    analytic(a, b) = sinpi(a) * sinpi(b) / (2pi^2)
    exact = [analytic(a, b) for a in xs, b in xs]
    prediction = [solution(a, b; dv = u(x, y)) for a in xs, b in xs]
    return sum(abs, prediction .- exact) / sum(abs, exact)
end

@testset "Adaptive loss 2D Poisson" begin
    rules = [
        ("GradientScale", (prob, w) -> GradientScaleAdaptiveLoss(prob, w; every = 100, inertia = 0.9), [1.0e3; ones(4)]),
        ("MiniMax", (prob, w) -> MiniMaxAdaptiveLoss(prob, w; every = 100), ones(5)),
        ("SoftAdapt", (prob, w) -> SoftAdaptAdaptiveLoss(prob, w; every = 100), ones(5)),
        ("ReLoBRaLo", (prob, w) -> ReLoBRaLoAdaptiveLoss(prob, w; every = 100, rng = Xoshiro(61)), ones(5)),
    ]
    for (name, callback, initial_weights) in rules
        @testset "$name" begin
            @test poisson_error(callback; initial_weights) < 0.4
        end
    end
end
