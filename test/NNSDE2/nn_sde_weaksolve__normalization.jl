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

function check_default_mass_and_amplitude_gradient(d, initial, lo, hi; check_gradient = true)
    alg = resolve_norm_loss_alg(nothing, lo, hi, initial)
    exact = cdf(d, hi) - cdf(d, lo)
    density(x, p) = p[1] * pdf(d, x)
    mass(a) = normalization_mass(density, lo, hi, [a], alg)
    @test mass(1.0) ≈ exact rtol = 1.0e-4
    return if check_gradient
        @test Zygote.gradient(mass, 1.0)[1] ≈ exact rtol = 1.0e-4
    end
end

@testset "default normalization rule tracks quantiles" begin
    mixture = MixtureModel(
        [Normal(0.375, 0.001), Normal(0.625, 0.001)], Categorical([0.5, 0.5])
    )
    check_default_mass_and_amplitude_gradient(mixture, mixture, 0.0, 1.0)

    # The rule is selected once from the initial condition, then reused as the density
    # contracts during training.
    check_default_mass_and_amplitude_gradient(
        Normal(0.375, 0.001), Normal(0.375, 0.1), 0.0, 1.0
    )
    capped = Normal(0.50048828125, 1.0e-6)
    check_default_mass_and_amplitude_gradient(capped, capped, 0.0, 1.0)

    err = try
        resolve_norm_loss_alg(nothing, 0.0, 1.0, Normal(0.5, 1.0e-20))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("pass `norm_loss_alg` explicitly", sprint(showerror, err))
end

@testset "default normalization rule supports Cauchy priors" begin
    truncated_cauchy = truncated(Cauchy(0.375, 0.01), 0.0, 1.0)
    check_default_mass_and_amplitude_gradient(
        truncated_cauchy, truncated_cauchy, 0.0, 1.0; check_gradient = false
    )
    cauchy = Cauchy(0.375, 0.001)
    check_default_mass_and_amplitude_gradient(
        cauchy, cauchy, 0.0, 1.0; check_gradient = false
    )
end

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
