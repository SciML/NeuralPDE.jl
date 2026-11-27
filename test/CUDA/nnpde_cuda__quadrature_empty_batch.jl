using CUDA, Integrals, NeuralPDE, SciMLBase, Test, Zygote

mutable struct EmptyBatchProbe <: SciMLBase.AbstractIntegralAlgorithm
    prototype::Any
end

function SciMLBase.solve(
        prob::SciMLBase.IntegralProblem, alg::EmptyBatchProbe; kwargs...
    )
    alg.prototype = prob.f(CUDA.zeros(Float64, 1, 0), prob.p)
    return (; u = ones(Float64, 1))
end

@testset "QuadratureTraining skips CUDA empty prototype batches" begin
    probe = EmptyBatchProbe(nothing)
    residuals = function (x, θ)
        isempty(x) && error("residual cannot evaluate an empty CUDA batch")
        return θ .* x
    end
    parameters = CUDA.fill(2.0, 1)
    strategy = QuadratureTraining(quadrature_alg = probe)
    loss = NeuralPDE.get_loss_function(
        parameters, residuals, [0.0], [1.0], Float64, strategy
    )

    @test only(loss(parameters)) == 1.0
    @test probe.prototype isa Vector{Float64}
    @test isempty(probe.prototype)
end

@testset "QuadratureTraining evaluates residuals on CUDA" begin
    residuals = (x, θ) -> θ .* x
    parameters = CUDA.fill(2.0, 1)
    strategy = QuadratureTraining(
        quadrature_alg = CubatureJLh(), reltol = 1.0e-8, abstol = 1.0e-8,
        maxiters = 10_000, batch = 16
    )
    loss = NeuralPDE.get_loss_function(
        parameters, residuals, [0.0], [1.0], Float64, strategy
    )

    @test only(loss(parameters)) ≈ 4 / 3
    gradient = only(Zygote.gradient(p -> only(loss(p)), parameters))
    @test only(Array(gradient)) ≈ 4 / 3
end
