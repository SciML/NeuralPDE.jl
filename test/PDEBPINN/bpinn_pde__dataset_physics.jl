using NeuralPDE, ModelingToolkit, SciMLBase, Lux, Random, Distributions,
    AdvancedHMC, MCMCChains, LogDensityProblems, ForwardDiff, Test
using DomainSets: Interval
ext = Base.get_extension(NeuralPDE, :NeuralPDEBPINNExt)
@parameters x p
@variables u(..)
Dx = Differential(x)
@named sys = PDESystem(
    (Dx^2)(u(x)) + p * u(x) ~ x,
    [u(0.0) ~ 1.0, Dx(u(1.0)) ~ 2.0],
    [x ∈ Interval(0.0, 1.0)], [x], [u(x)], [p];
    initial_conditions = Dict(p => 1.0)
)
disc = PhysicsInformedNN(
    Lux.Chain(Lux.Dense(1, 2, tanh), Lux.Dense(2, 1)),
    GridTraining(0.25); param_estim = true, rng = Xoshiro(100)
)
md = NeuralPDE.pinn_metadata(SciMLBase.discretize(sys, disc))
xs = [0.2, 0.7]
data = hcat(zeros(2), xs)
network_fns = [(X, θ) -> reshape(θ[1] .* X[1, :] .^ 2, 1, :)]
ll = ext.build_physics_at_points(
    sys, [data], [data, data], network_fns, [1], md,
    [0.3], [0.4, 0.5]
)
function reference(θ)
    a, p = θ
    return sum(logpdf(Normal(0, 0.3), 2a + p * a * x^2 - x) for x in xs) +
        2logpdf(Normal(0, 0.4), -1.0) + 2logpdf(Normal(0, 0.5), 2a - 2)
end
θ = [0.7, 1.2]
@testset "Dataset-point physics analytic reference" begin
    @test ll(θ) ≈ reference(θ) rtol = 1.0e-8
    @test ForwardDiff.gradient(ll, θ) ≈ ForwardDiff.gradient(reference, θ) rtol = 1.0e-7
end
