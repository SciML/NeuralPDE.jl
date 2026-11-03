using NeuralPDE, Integrals, Test

# https://github.com/SciML/NeuralPDE.jl/issues/1215
# A Lorentzian centred at 0.5, whose mass over [0, 1] is elementary. At c = 0.01 the peak
# is narrower than one quadrature panel, so a rule that never refines misses it.
lorentz(x, p) = (p[1] / pi) / ((x - 0.5)^2 + p[1]^2)
exact_mass(c) = (2 / pi) * atan(0.5 / c)

@testset "normalization mass with the default rule" begin
    for c in (0.1, 0.01)
        @test NeuralPDE.normalization_mass(lorentz, 0.0, 1.0, [c], nothing) ≈
            exact_mass(c) rtol = 1.0e-4
    end
end

@testset "normalization mass with an adaptive rule" begin
    for c in (0.1, 0.01)
        @test NeuralPDE.normalization_mass(lorentz, 0.0, 1.0, [c], QuadGKJL()) ≈
            exact_mass(c) rtol = 1.0e-4
    end
end
