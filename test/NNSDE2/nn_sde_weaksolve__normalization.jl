using NeuralPDE, Integrals, Zygote, Test
using NeuralPDE: normalization_mass, resolve_norm_loss_alg

# https://github.com/SciML/NeuralPDE.jl/issues/1215
# A Lorentzian centred at 0.5, whose mass over [0, 1] is elementary. At c = 0.01 the peak
# is narrower than one quadrature panel, so a rule that never refines misses it.
lorentz(x, p) = (p[1] / pi) / ((x - 0.5)^2 + p[1]^2)
exact_mass(c) = (2 / pi) * atan(0.5 / c)
exact_dmass(c) = -1 / (pi * (c^2 + 0.25))

@testset "normalization mass with the default rule" begin
    alg = resolve_norm_loss_alg(nothing)
    for c in (0.1, 0.01)
        @test normalization_mass(lorentz, 0.0, 1.0, [c], alg) ≈ exact_mass(c) rtol = 1.0e-4
    end
end

@testset "normalization mass with an adaptive rule" begin
    for c in (0.1, 0.01)
        @test normalization_mass(lorentz, 0.0, 1.0, [c], QuadGKJL()) ≈
            exact_mass(c) rtol = 1.0e-4
    end
end

# the loss is differentiated, so the quadrature has to survive Zygote
@testset "normalization mass differentiates" begin
    alg = resolve_norm_loss_alg(nothing)
    for c in (0.1, 0.01)
        g = Zygote.gradient(x -> normalization_mass(lorentz, 0.0, 1.0, [x], alg), c)[1]
        @test g ≈ exact_dmass(c) rtol = 1.0e-3
    end
end
