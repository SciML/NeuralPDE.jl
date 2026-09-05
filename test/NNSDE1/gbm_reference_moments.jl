using Test, Distributions, Statistics

include("../helpers/gbm_reference.jl")

@testset "GBM independent-reference squared errors" begin
    u0, alpha, beta = 0.5, 1.2, 1.1
    for t in (0.0, 0.02, 0.4, 1.0), prediction in (-2.0, 0.0, 0.5, 3.0), n in (1, 2, 2000)
        reference = LogNormal(log(u0) + (alpha - beta^2 / 2) * t, beta * sqrt(t))
        expected = var(reference) / n + abs2(mean(reference) - prediction)
        @test gbm_reference_squared_error(u0, alpha, beta, t, prediction, n) ≈ expected
    end
end
