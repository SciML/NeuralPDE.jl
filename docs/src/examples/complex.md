# Complex Equations with PINNs

NeuralPDE supports training PINNs with complex differential equations. This example will demonstrate how to use it for [`NNODE`](@ref). Let us consider a system of [bloch equations](https://en.wikipedia.org/wiki/Bloch_equations) [^1]. Note [`QuadratureTraining`](@ref) cannot be used with complex equations due to current limitations of computing quadratures.

As the input to this neural network is time which is real, we need to initialize the parameters of the neural network with complex values for it to output and train with complex values.

```@example complex
using Random, NeuralPDE
using OrdinaryDiffEq
using Lux, OptimizationOptimisers
using Plots
rng = Random.default_rng()
Random.seed!(100)

function bloch_equations(u, p, t)
    Ω, Δ, Γ = p
    γ = Γ / 2
    ρ₁₁, ρ₂₂, ρ₁₂, ρ₂₁ = u
    d̢ρ = [im * Ω * (ρ₁₂ - ρ₂₁) + Γ * ρ₂₂;
           -im * Ω * (ρ₁₂ - ρ₂₁) - Γ * ρ₂₂;
           -(γ + im * Δ) * ρ₁₂ - im * Ω * (ρ₂₂ - ρ₁₁);
           conj(-(γ + im * Δ) * ρ₁₂ - im * Ω * (ρ₂₂ - ρ₁₁))]
    return d̢ρ
end

u0 = zeros(ComplexF64, 4)
u0[1] = 1.0
time_span = (0.0, 2.0)
parameters = [2.0, 0.0, 1.0]

problem = ODEProblem(bloch_equations, u0, time_span, parameters)

chain = Lux.Chain(
    Lux.Dense(1, 16, tanh;
        init_weight = (rng, a...) -> Lux.kaiming_normal(rng, ComplexF64, a...)),
    Lux.Dense(
        16, 4; init_weight = (rng, a...) -> Lux.kaiming_normal(rng, ComplexF64, a...))
)
ps, st = Lux.setup(rng, chain)

opt = OptimizationOptimisers.Adam(0.01)
ground_truth = solve(problem, Tsit5(), saveat = 0.01)
alg = NNODE(chain, opt, ps; strategy = StochasticTraining(500))
sol = solve(problem, alg, verbose = false, maxiters = 5000, saveat = 0.01)
```

Now, let's plot the predictions.

`u1`:

```@example complex
plot(sol.t, real.(reduce(hcat, sol.u)[1, :]));
plot!(ground_truth.t, real.(reduce(hcat, ground_truth.u)[1, :]))
```

```@example complex
plot(sol.t, imag.(reduce(hcat, sol.u)[1, :]));
plot!(ground_truth.t, imag.(reduce(hcat, ground_truth.u)[1, :]))
```

`u2`:

```@example complex
plot(sol.t, real.(reduce(hcat, sol.u)[2, :]));
plot!(ground_truth.t, real.(reduce(hcat, ground_truth.u)[2, :]))
```

```@example complex
plot(sol.t, imag.(reduce(hcat, sol.u)[2, :]));
plot!(ground_truth.t, imag.(reduce(hcat, ground_truth.u)[2, :]))
```

`u3`:

```@example complex
plot(sol.t, real.(reduce(hcat, sol.u)[3, :]));
plot!(ground_truth.t, real.(reduce(hcat, ground_truth.u)[3, :]))
```

```@example complex
plot(sol.t, imag.(reduce(hcat, sol.u)[3, :]));
plot!(ground_truth.t, imag.(reduce(hcat, ground_truth.u)[3, :]))
```

`u4`:

```@example complex
plot(sol.t, real.(reduce(hcat, sol.u)[4, :]));
plot!(ground_truth.t, real.(reduce(hcat, ground_truth.u)[4, :]))
```

```@example complex
plot(sol.t, imag.(reduce(hcat, sol.u)[4, :]));
plot!(ground_truth.t, imag.(reduce(hcat, ground_truth.u)[4, :]))
```

We can see it is able to learn the real parts of `u1`, `u2` and imaginary parts of `u3`, `u4`.

[^1]: https://steck.us/alkalidata/
