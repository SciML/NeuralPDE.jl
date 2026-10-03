# Bayesian PINNs for PDEs

This tutorial solves a scalar ODE written as a ModelingToolkit `PDESystem` with a
Bayesian physics-informed neural network. The [`BayesianPINN`](@ref) discretizer
lowers the system through the same residual `System` as [`PhysicsInformedNN`](@ref);
[`ahmc_bayesian_pinn_pde`](@ref) then treats each residual batch as a Gaussian
likelihood and samples the network weights with AdvancedHMC.

!!! note "Loading the Bayesian PINN extension"

    Load `AdvancedHMC`, `MCMCChains` and `LogDensityProblems` alongside `NeuralPDE`.

```@example bpinn_pde
using NeuralPDE, Lux, ModelingToolkit, AdvancedHMC, MCMCChains, LogDensityProblems
using Statistics, Random, MonteCarloMeasurements
using DomainSets: Interval

Random.seed!(100)

@parameters t
@variables u(..)
Dt = Differential(t)
eq = Dt(u(t)) - cospi(2t) ~ 0
bcs = [u(0.0) ~ 0.0]
domains = [t ∈ Interval(0.0, 2.0)]
@named pde_system = PDESystem(eq, bcs, domains, [t], [u(t)])

chain = Chain(Dense(1, 6, tanh), Dense(6, 1))
discretization = BayesianPINN(chain, GridTraining(0.01))

sol = ahmc_bayesian_pinn_pde(
    pde_system, discretization;
    draw_samples = 200, bcstd = [0.01], phystd = [0.01],
    priorsNNw = (0.0, 1.0), saveats = [1 / 50.0]
)

analytic(t) = sinpi(2t) / (2pi)
ts = vec(sol.timepoints[1])
u_predict = pmean(sol.ensemblesol[1])
mean(abs, u_predict .- analytic.(ts))
```

The returned [`BPINNsolution`](@ref) stores the MCMC chain in `sol.original`, the
ensemble prediction as `MonteCarloMeasurements.Particles` in `sol.ensemblesol`, and
(when `param_estim = true`) posterior estimates of the PDE parameters in
`sol.estimated_de_params`.
