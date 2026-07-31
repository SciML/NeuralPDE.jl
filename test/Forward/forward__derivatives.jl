using NeuralPDE
using Test

@testset "derivatives" begin
    using DomainSets, Lux, Random, Zygote, ComponentArrays

    chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))
    init_params = Lux.initialparameters(Random.default_rng(), chain) |>
        ComponentArray{Float64}

    eltypeθ = eltype(init_params)
    phi = NeuralPDE.Phi(chain)
    derivative = NeuralPDE.numeric_derivative

    u_ = (cord, θ, phi) -> sum(phi(cord, θ))

    phi([1, 2], init_params)

    phi_ = (p) -> phi(p, init_params)[1]
    dphi = Zygote.gradient(phi_, [1.0, 2.0])

    eps_x = NeuralPDE.get_ε(2, 1, Float64, 1)
    eps_y = NeuralPDE.get_ε(2, 2, Float64, 1)

    dphi_x = derivative(phi, u_, [1.0, 2.0], [eps_x], 1, init_params)
    dphi_y = derivative(phi, u_, [1.0, 2.0], [eps_y], 1, init_params)

    #first order derivatives
    @test isapprox(dphi[1][1], dphi_x, atol = 1.0e-8)
    @test isapprox(dphi[1][2], dphi_y, atol = 1.0e-8)

    eps_x = NeuralPDE.get_ε(2, 1, Float64, 2)
    eps_y = NeuralPDE.get_ε(2, 2, Float64, 2)

    hess_phi = Zygote.hessian(phi_, [1, 2])

    dphi_xx = derivative(phi, u_, [1.0, 2.0], [eps_x, eps_x], 2, init_params)
    dphi_xy = derivative(phi, u_, [1.0, 2.0], [eps_x, eps_y], 2, init_params)
    dphi_yy = derivative(phi, u_, [1.0, 2.0], [eps_y, eps_y], 2, init_params)

    #second order derivatives
    @test isapprox(hess_phi[1], dphi_xx, atol = 4.0e-5)
    @test isapprox(hess_phi[2], dphi_xy, atol = 4.0e-5)
    @test isapprox(hess_phi[4], dphi_yy, atol = 4.0e-5)
end
