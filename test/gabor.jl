using LinearAlgebra
using Plots

function gabor(x, γ, μ, ω, ϕ)
    mean_offset = x .- μ
    exp(-γ/2 .* dot(mean_offset, mean_offset)) .* sin(ω .* x .+ ϕ)
end

xs = range(-3, 3, 100)
γ = 1.0
μ = 0.0
ω = 1.0
ϕ = 0.0

ys = gabor.(xs, γ, μ, ω, ϕ)

plot(xs, ys)
