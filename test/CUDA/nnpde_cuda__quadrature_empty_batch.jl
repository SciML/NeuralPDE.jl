using CUDA, NeuralPDE, SciMLBase, Test

mutable struct EmptyBatchProbe <: SciMLBase.AbstractIntegralAlgorithm
    prototype::Any
end

function SciMLBase.solve(
        prob::SciMLBase.IntegralProblem, alg::EmptyBatchProbe; kwargs...
    )
    alg.prototype = prob.f(CUDA.zeros(Float64, 1, 0), prob.p)
    return (; u = CUDA.ones(Float64, 1))
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

    @test only(Array(loss(parameters))) == 1.0
    @test probe.prototype isa CuArray{Float64, 1}
    @test isempty(probe.prototype)
end
