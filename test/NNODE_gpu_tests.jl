using Test
using Random, NeuralPDE
using OrdinaryDiffEq, Statistics
import Lux, OptimizationOptimisers
using OptimizationOptimJL, LineSearches
using LuxCUDA
using ComponentArrays
using LuxCUDA.CUDA: CuArray
using StaticArrays

rng = Random.default_rng()
Random.seed!(100)
const gpud = Lux.gpu_device()

@testset "Vector" begin
    # Run a solve on vectors
    println("Vector")
    linear = (u, p, t) -> vcat(cos.(t))
    tspan = (0.0, 1.0)
    u0 = [0.0]
    prob = ODEProblem(linear, u0, tspan)
    luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))

    ps = Lux.setup(rng, luxchain)[1] |> ComponentArray |> gpud .|> Float64

    opt = OptimizationOptimJL.BFGS()
    sol = solve(prob, NNODE(luxchain, opt; device = gpud), dt = 1 / 20.0, abstol = 1e-10,
        verbose = true, maxiters = 200)

    @test sol(0.5) isa CuArray
    @test sol.k isa SciMLBase.OptimizationSolution
end

@testset "Training Strategies" begin
    @testset "WeightedIntervalTraining" begin
        println("WeightedIntervalTraining")
        function f(u, p, t)
            SVector{2}(p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2])
        end
        p = SVector{4}(1.5, 1.0, 3.0, 1.0)
        u0 = SVector{2}(1.0, 1.0)
        prob_oop = ODEProblem{false}(f, u0, (0.0, 3.0), p)
        true_sol = solve(prob_oop, Tsit5(), saveat = 0.01)
        func = Lux.σ
        N = 12
        chain = Lux.Chain(
            Lux.Dense(1, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func),
            Lux.Dense(N, N, func), Lux.Dense(N, length(u0)))
        ps = Lux.setup(rng, chain)[1] |> ComponentArray .|> Float64 |> gpud
        opt = OptimizationOptimisers.Adam(0.01)
        weights = [0.7, 0.2, 0.1]
        points = 200
        alg = NNODE(chain, opt; autodiff = false,
            strategy = NeuralPDE.WeightedIntervalTraining(weights, points), device = gpud)
        out = CuArray(rand(2, 100))
        p_ = CuArray(p)
        t = CuArray(rand(100))
        du = similar(out)
        NeuralPDE.gpu_broadcast(
            prob_oop.f, du, out, p_, t; workgroupsize = 64, ndrange = 100)
        sol = solve(prob_oop, alg, verbose = false, maxiters = 5000, saveat = 0.01)
        @test abs(mean(sol) - mean(true_sol)) < 0.2
    end

    linear = (u, p, t) -> cos(2pi * t)
    linear_analytic = (u, p, t) -> (1 / (2pi)) * sin(2pi * t)
    tspan = (0.0, 1.0)
    dt = (tspan[2] - tspan[1]) / 99
    ts = collect(tspan[1]:dt:tspan[2])
    prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), 0.0, (0.0, 1.0))
    opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))
    u_analytical(x) = (1 / (2pi)) .* sin.(2pi .* x)

    @testset "GridTraining" begin
        println("GridTraining")
        luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
        ps = Lux.setup(rng, luxchain)[1] |> ComponentArray .|> Float64 |> gpud
        (u_, t_) = (u_analytical(ts), ts)
        function additional_loss(phi, θ)
            return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
        end
        alg1 = NNODE(luxchain, opt, ps; strategy = GridTraining(0.01),
            additional_loss = additional_loss)
        sol1 = solve(prob, alg1, verbose = false, abstol = 1e-8, maxiters = 500)
        @test sol1.errors[:l2] < 0.5
    end

    @testset "StochasticTraining" begin
        println("StochasticTraining")
        luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
        ps = Lux.setup(rng, luxchain)[1] |> ComponentArray .|> Float64 |> gpud
        (u_, t_) = (u_analytical(ts), ts)
        function additional_loss(phi, θ)
            return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
        end
        alg1 = NNODE(luxchain, opt, ps; strategy = StochasticTraining(1000),
            additional_loss = additional_loss)
        sol1 = solve(prob, alg1, verbose = false, abstol = 1e-8, maxiters = 500)
        @test sol1.errors[:l2] < 0.5
    end
end

### MWE:
using Random, NeuralPDE
using OrdinaryDiffEq
using Lux, OptimizationOptimisers
using LuxCUDA, ComponentArrays

rng = Random.default_rng()
Random.seed!(100)
const gpud = Lux.gpu_device()

linear = (u, p, t) -> cos.(2pi * t)
tspan = (0.0, 1.0)
u0 = 0.0
prob = ODEProblem(linear, u0, tspan)
luxchain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
ps = Lux.setup(rng, luxchain)[1] |> ComponentArray |> gpud .|> Float64
opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))

sol = solve(prob, NNODE(luxchain, opt, ps; strategy = GridTraining(0.01), device = gpud),
    verbose = true, maxiters = 200)

sol = solve(
    prob, NNODE(luxchain, opt, ps; strategy = StochasticTraining(100), device = gpud),
    verbose = true, maxiters = 200)

# GPU64 function 

function g!(f, buf, args..)
    buf .= f(args...)
end

function rrule(::typeof(g!), ...)
    gbuf = similar(buf)
    function g_pullback(Del)
        fwd, pb = Zygote.pullback(args...) do
            f(args...)
        end
        gbuf .= pb(Del)
    end
    g!(...), g_pullback
end


## MWE:
using Random, NeuralPDE
using OrdinaryDiffEq
using Lux, OptimizationOptimisers
using LuxCUDA, ComponentArrays, StaticArrays

rng = Random.default_rng()
Random.seed!(100)
const gpud = Lux.gpu_device()

function f(u, p, t)
    SVector{2}(p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[4] * u[1] * u[2])
end
p = SVector{4}(1.5, 1.0, 3.0, 1.0)
u0 = SVector{2}(1.0, 1.0)
prob_oop = ODEProblem{false}(f, u0, (0.0, 3.0), p)
func = Lux.σ
N = 12
chain = Lux.Chain(
    Lux.Dense(1, N, func), Lux.Dense(N, N, func), Lux.Dense(N, N, func),
    Lux.Dense(N, N, func), Lux.Dense(N, length(u0)))
opt = OptimizationOptimisers.Adam(0.01)
weights = [0.7, 0.2, 0.1]
points = 200
alg = NNODE(chain, opt; autodiff = false,
    strategy = NeuralPDE.GridTraining(0.1), device = gpud)
sol = solve(prob_oop, alg, verbose = true, maxiters = 10, saveat = 0.01)

# This works
out = CuArray(rand(2, 100))
p_ = CuArray(p)
t = CuArray(rand(100))
du = similar(out)
NeuralPDE.gpu_broadcast(prob_oop.f, du, out, p_, t; workgroupsize = 64, ndrange = 100)
du
