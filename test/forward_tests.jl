using Flux
println("forward_tests")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
println("Starting Soon!")
using SciMLBase
import ModelingToolkit: Interval, infimum, supremum

@testset "ODE" begin
    @parameters x
    @variables u(..)

    Dx = Differential(x)
    eq = Dx(u(x)) ~ 0.
    bcs = [u(0.) ~ u(0.)]
    domains = [x ∈ Interval(0.0,1.0)]
    chain = FastChain((x,p) -> x.^2)

    chain([1],Float32[])
    strategy_ = NeuralPDE.GridTraining(0.1)
    discretization = NeuralPDE.PhysicsInformedNN(chain,strategy_)
    @named pde_system = PDESystem(eq,bcs,domains,[x],[u(x)])
    prob = NeuralPDE.discretize(pde_system,discretization)

    train_data =prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents[1].train_set
    inner_loss =prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents[1].loss_function

    dudx(x) = @. 2*x
    @test inner_loss(train_data, Float32[]) ≈ dudx(train_data) rtol = 0.001
end

@testset "derivatives" begin
    chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
    initθ = Float64.(DiffEqFlux.initial_params(chain))

    eltypeθ = eltype(initθ)
    parameterless_type_θ = DiffEqBase.parameterless_type(initθ)
    phi = NeuralPDE.get_phi(chain,parameterless_type_θ)
    derivative = NeuralPDE.get_numeric_derivative()

    u_ = (cord, θ, phi)->sum(phi(cord, θ))

    phi([1,2], initθ)

    phi_ = (p) -> phi(p, initθ)[1]
    dphi = Zygote.gradient(phi_,[1.,2.])

    function get_ε(dim, der_num,eltypeθ)
        epsilon = cbrt(eps(eltypeθ))
        ε = zeros(eltypeθ, dim)
        ε[der_num] = epsilon
        ε
    end

    eps_x = get_ε(2, 1,Float64)
    eps_y = get_ε(2, 2,Float64)

    dphi_x = derivative(phi,u_,[1.,2.],[eps_x],1,initθ)
    dphi_y = derivative(phi,u_,[1.,2.],[eps_y],1,initθ)

    #first order derivatives
    @test isapprox(dphi[1][1], dphi_x, atol=1e-8)
    @test isapprox(dphi[1][2], dphi_y, atol=1e-8)

    dphi_x = derivative(phi,u_,[1.,2.],[[ 0.0049215667, 0.0]],1,initθ)
    dphi_y = derivative(phi,u_,[1.,2.],[[0.0,  0.0049215667]],1,initθ)

    hess_phi = Zygote.hessian(phi_,[1,2])

    dphi_xx = derivative(phi,u_,[1.,2.],[eps_x,eps_x],2,initθ)
    dphi_xy = derivative(phi,u_,[1.,2.],[eps_x,eps_y],2,initθ)
    dphi_yy = derivative(phi,u_,[1.,2.],[eps_y,eps_y],2,initθ)

    #second order derivatives
    @test isapprox(hess_phi[1], dphi_xx, atol=1e-5)
    @test isapprox(hess_phi[2], dphi_xy, atol=1e-5)
    @test isapprox(hess_phi[4], dphi_yy, atol=1e-5)
end
