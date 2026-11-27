using AdvancedHMC, Distributions, ForwardDiff, LogDensityProblems, Lux, MCMCChains,
    ModelingToolkit, NeuralPDE, Random, SciMLBase, Test
using DomainSets: Interval
using LinearAlgebra: Diagonal
using MonteCarloMeasurements: pmean

@testset "Fixed-coefficient Bayesian PDE observations" begin
    @parameters x t
    @variables u(..)
    @named sys = PDESystem(
        Differential(t)(u(x, t)) ~ 0, [u(0.0, t) ~ 0.0],
        [x ∈ Interval(0.0, 1.0), t ∈ Interval(0.0, 1.0)], [x, t], [u(x, t)]
    )

    signal = 2.0
    xs = [0.5, 1.0]
    σ = 0.1
    data = hcat(signal .* xs, xs, zeros(length(xs)))
    shifted = hcat((signal + 1) .* xs, xs, zeros(length(xs)))
    chain = Lux.Chain(Lux.Dense(2, 1; use_bias = false))
    strategy = GridTraining(0.5)
    prob = SciMLBase.discretize(sys, PhysicsInformedNN(chain, strategy; rng = Xoshiro(42)))
    ext = Base.get_extension(NeuralPDE, :NeuralPDEBPINNExt)
    function target(observations)
        return ext.PDELogTargetDensity(
            2, prob, (), Float64[], [MvNormal(zeros(2), Diagonal(ones(2)))], 0,
            [observations], [σ], [(X, θ) -> transpose(θ) * X], [2], Float64[],
            nothing, nothing
        )
    end
    observed_target, shifted_target = target(data), target(shifted)
    observed_logdensity(θ) = LogDensityProblems.logdensity(observed_target, θ)
    shifted_logdensity(θ) = LogDensityProblems.logdensity(shifted_target, θ)
    reference(observations, θ) = sum(logpdf(Normal(θ[1] * x, σ), y) for (y, x, _) in eachrow(observations))
    θ = [0.5, 0.25]
    @test observed_logdensity(θ) - shifted_logdensity(θ) ≈ reference(data, θ) - reference(shifted, θ)
    gradient_shift = ForwardDiff.gradient(observed_logdensity, θ) - ForwardDiff.gradient(shifted_logdensity, θ)
    @test gradient_shift ≈ [-sum(abs2, xs) / σ^2, 0.0]

    function fit(dataset)
        Random.seed!(42)
        disc = BayesianPINN(chain, strategy; dataset, rng = Xoshiro(42))
        sol = ahmc_bayesian_pinn_pde(
            sys, disc; draw_samples = 1200, numensemble = 400, pretrain_iters = 0,
            l2std = [σ], phystd = [0.5], bcstd = [0.5], priorsNNw = (0.0, 1.0),
            saveats = [0.5, 0.5]
        )
        idx = findfirst(==((1.0, 0.0)), Tuple.(eachcol(only(sol.timepoints))))
        return pmean(only(sol.ensemblesol))[idx]
    end

    # Physics constrains only the time weight, so the spatial weight has a Gaussian posterior.
    data_precision = sum(abs2, xs) / σ^2
    posterior_mean = signal * data_precision / (1 + data_precision)
    posterior_std = inv(sqrt(1 + data_precision))
    @test abs(fit((nothing, nothing)) - signal) > signal / 2
    @test abs(fit(([data], nothing)) - posterior_mean) < posterior_std
end
