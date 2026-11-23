using NeuralPDE, Distributions, Integrals, Zygote, Test
using NeuralPDE: normalization_mass, resolve_norm_loss_alg

# A Gaussian of amplitude p[1] centred at p[2], narrow next to the interval it is
# integrated over. Its mass and the derivative of that mass in the amplitude are both the
# enclosed probability.
scaled_normal(x, p) = p[1] * pdf(Normal(p[2], 0.01), x)
enclosed(mu) = cdf(Normal(mu, 0.01), 4.0) - cdf(Normal(mu, 0.01), -4.0)

# A Lorentzian centred at 0.5, whose mass over [0, 1] is elementary.
lorentz(x, p) = (p[1] / pi) / ((x - 0.5)^2 + p[1]^2)
exact_lorentz_mass(c) = (2 / pi) * atan(0.5 / c)

@testset "narrow peak with the default rule" begin
    for mu in (0.5, 1.0, -3.1)
        alg = resolve_norm_loss_alg(nothing, -4.0, 4.0, Normal(mu, 0.01))
        @test normalization_mass(scaled_normal, -4.0, 4.0, [1.0, mu], alg) ≈
            enclosed(mu) rtol = 1.0e-4
        g = Zygote.gradient(
            a -> normalization_mass(scaled_normal, -4.0, 4.0, [a, mu], alg), 1.0
        )[1]
        @test g ≈ enclosed(mu) rtol = 1.0e-4
    end
end

@testset "narrow peak with the caller's rule" begin
    alg = resolve_norm_loss_alg(QuadGKJL(), 0.0, 1.0, Normal(0.5, 0.01))
    @test alg isa QuadGKJL
    for c in (0.1, 0.01)
        @test normalization_mass(lorentz, 0.0, 1.0, [c], alg) ≈
            exact_lorentz_mass(c) rtol = 1.0e-4
    end
end
