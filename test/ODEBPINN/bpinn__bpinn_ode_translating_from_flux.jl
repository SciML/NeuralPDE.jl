using NeuralPDE
using Test

@testset "BPINN ODE: Translating from Flux" begin
    using MCMCChains, Distributions, OrdinaryDiffEq, OptimizationOptimisers, Lux,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, ComponentArrays, MonteCarloMeasurements
    import Flux

    Random.seed!(100)

    linear_analytic = (u0, p, t) -> u0 + sin(2 * π * t) / (2 * π)
    linear = (u, p, t) -> cos(2 * π * t)
    tspan = (0.0, 2.0)
    u0 = 0.0
    prob = ODEProblem(ODEFunction(linear, analytic = linear_analytic), u0, tspan)
    p = prob.p

    # Numerical and Analytical Solutions: testing ahmc_bayesian_pinn_ode()
    ta = range(tspan[1], tspan[2], length = 300)
    u = [linear_analytic(u0, nothing, ti) for ti in ta]
    x̂ = collect(Float64, Array(u) + 0.02 * randn(size(u)))
    time = vec(collect(Float64, ta))
    physsol1 = [linear_analytic(prob.u0, p, time[i]) for i in eachindex(time)]

    # testing points for solve() call must match saveat(1/50.0) arg
    ta0 = range(tspan[1], tspan[2], length = 101)
    u1 = [linear_analytic(u0, nothing, ti) for ti in ta0]
    x̂1 = collect(Float64, Array(u1) + 0.02 * randn(size(u1)))
    time1 = vec(collect(Float64, ta0))
    physsol0_1 = [linear_analytic(prob.u0, p, time1[i]) for i in eachindex(time1)]
    chainflux = Flux.Chain(Flux.Dense(1, 7, tanh), Flux.Dense(7, 1)) |> Flux.f64
    fh_mcmc_chain, fhsamples,
        fhstats = ahmc_bayesian_pinn_ode(
        prob, chainflux, draw_samples = 2500
    )
    alg = BNNODE(chainflux, draw_samples = 2500)
    @test alg.chain isa AbstractLuxLayer
end
