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

@testset "Observational dataset validation" begin
    for invalid in ([data, data], [data[:, 1:1]])
        @test_throws ArgumentError ext.build_physics_at_points(
            sys, invalid, nothing, network_fns, [1], md, [0.3], [0.4, 0.5]
        )
    end
    D = Dict((Dx^2)(u(x)) => Symbolics.variable(:diff_1))
    quad = ext.build_data_quadrature(sys, D, [data], network_fns, [1], md)
    for stds in (Float64[], [0.3, 0.4])
        @test_throws ArgumentError quad(θ, stds)
    end
    for invalid in (Matrix{Float64}[], [data, data], [data[:, 1:1]])
        @test_throws ArgumentError ext.build_data_quadrature(
            sys, D, invalid, network_fns, [1], md
        )
    end
end

@testset "Data quadrature coordinate alignment" begin
    @parameters x t
    @variables u(..) v(..)
    Dt = Differential(t)
    for v_args in ([x, t], [t])
        @named coupled = PDESystem(
            [
                Dt(u(x, t)) ~ v(v_args...) + cos(x) + sin(t),
                Dt(v(v_args...)) ~ u(x, t) + sin(x) + cos(t),
            ],
            [u(x, 0.0) ~ 0.0, v(v_args[1:(end - 1)]..., 0.0) ~ 0.0],
            [x ∈ Interval(0.0, 1.0), t ∈ Interval(0.0, 1.0)],
            [x, t], [u(x, t), v(v_args...)]
        )
        disc = PhysicsInformedNN(
            [Lux.Chain(Lux.Dense(2, 1)), Lux.Chain(Lux.Dense(length(v_args), 1))],
            GridTraining(0.5); rng = Xoshiro(101)
        )
        md = NeuralPDE.pinn_metadata(SciMLBase.discretize(coupled, disc))
        data = [0.0 0.2 0.3; 0.0 0.4 0.5]
        shifted = copy(data)
        shifted[:, 3] .+= 0.1
        D = Dict(
            Dt(u(x, t)) => Symbolics.variable(:diff_u),
            Dt(v(v_args...)) => Symbolics.variable(:diff_v)
        )
        message = length(v_args) == 2 ? "matching coordinates" : "network's arguments"
        @test_throws message ext.build_data_quadrature(
            coupled, D, [data, shifted], [], [3, length(v_args) + 1], md
        )
        if length(v_args) == 2
            network_fns = [(X, θ) -> reshape(θ[1] .* X[2, :], 1, :) for _ in 1:2]
            quad = ext.build_data_quadrature(
                coupled, D, [data, data], network_fns, [3, 3], md
            )
            reference = sum(
                logpdf(Normal(0, 0.3), 0.5 - cos(row[2]) - sin(row[3])) +
                    logpdf(Normal(0, 0.3), 0.5 - sin(row[2]) - cos(row[3]))
                    for row in eachrow(data)
            )
            @test quad(fill(0.5, 6), [0.3, 0.3]) ≈ reference rtol = 1.0e-8
        end
    end
end
