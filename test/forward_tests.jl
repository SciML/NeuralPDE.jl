using Flux
println("forward_tests")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
println("Starting Soon!")
using SciMLBase
using ForwardDiff
using DomainSets
import ModelingToolkit: Interval

@testset "ODE" begin
    @parameters x
    @variables u(..)

    Dx = Differential(x)
    eq = Dx(u(x)) ~ 0.
    bcs = [u(0.) ~ u(0.)]
    domains = [x ∈ Interval(0.0,1.0)]
    chain = FastChain((x,p) -> x.^2)

    chain([1],Float64[])
    strategy_ = NeuralPDE.GridTraining(0.1)
    discretization = NeuralPDE.PhysicsInformedNN(chain,strategy_;init_params = Float64[])
    @named pde_system = PDESystem(eq,bcs,domains,[x],[u(x)])
    prob = NeuralPDE.discretize(pde_system,discretization)

    train_data =prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents[1].train_set
    inner_loss =prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents[1].loss_function

    dudx(x) = @. 2*x
    @test inner_loss(train_data, Float64[]) ≈ dudx(train_data) rtol = 1e-8
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

    #first order numeric derivatives
    @test isapprox(dphi[1][1], dphi_x, atol=1e-9)
    @test isapprox(dphi[1][2], dphi_y, atol=1e-9)

    dphi_x = derivative(phi,u_,[1.,2.],[[ 0.0049215667, 0.0]],1,initθ)
    dphi_y = derivative(phi,u_,[1.,2.],[[0.0,  0.0049215667]],1,initθ)

    hess_phi = Zygote.hessian(phi_,[1,2])

    dphi_xx = derivative(phi,u_,[1.,2.],[eps_x,eps_x],2,initθ)
    dphi_xy = derivative(phi,u_,[1.,2.],[eps_x,eps_y],2,initθ)
    dphi_yy = derivative(phi,u_,[1.,2.],[eps_y,eps_y],2,initθ)

    #second order numeric derivatives
    @test isapprox(hess_phi[1], dphi_xx, atol=1e-5)
    @test isapprox(hess_phi[2], dphi_xy, atol=1e-5)
    @test isapprox(hess_phi[4], dphi_yy, atol=1e-5)

    #second order derivatives AD
    dphi_xad = eval(NeuralPDE.parser_derivative(phi,[:x,:y],[1]))
    dphi_yad = eval(NeuralPDE.parser_derivative(phi,[:x,:y],[2]))
    dphi_xxad = eval(NeuralPDE.parser_derivative(phi,[:x,:y],[1,1]))
    dphi_yyad = eval(NeuralPDE.parser_derivative(phi,[:x,:y],[2,2]))
    dphi_xyad = eval(NeuralPDE.parser_derivative(phi,[:x,:y],[1,2]))
    dphi_yxad = eval(NeuralPDE.parser_derivative(phi,[:x,:y],[2,1]))

    @test isapprox(hess_phi[1], dphi_xxad(initθ,1,2), atol=1e-9)
    @test isapprox(hess_phi[4], dphi_yyad(initθ,1,2), atol=1e-9)
    @test isapprox(hess_phi[2], dphi_xyad(initθ,1,2), atol=1e-9)
    @test isapprox(hess_phi[2], dphi_yxad(initθ,1,2), atol=1e-9)
end


@testset "Integral" begin
    #semi-infinite intervals
    @parameters x
    @variables u(..)
    I = Integral(x in ClosedInterval(0, Inf))
    eq = I(u(x)) ~ 0
    bcs = [u(1.) ~ exp(1)/(exp(2) + 3)]
    domains = [x ∈ Interval(1.0, 2.0)]
    chain = FastChain((x,p) -> exp.(x) ./ (exp.(2 .*x) .+ 3))
    chain([1],Float64[])
    strategy_ = NeuralPDE.GridTraining(0.1)
    discretization = NeuralPDE.PhysicsInformedNN(chain,strategy_;init_params = Float64[])
    @named pde_system = PDESystem(eq,bcs,domains,[x],[u(x)])
    sym_prob = SciMLBase.symbolic_discretize(pde_system, discretization)
    prob = NeuralPDE.discretize(pde_system,discretization)
    inner_loss =prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents[1].loss_function
    exact_u = π/(3*sqrt(3))
    @test  inner_loss(ones(1,1), Float64[])[1] ≈ exact_u  rtol = 1e-5


    #infinite intervals
    @parameters x
    @variables u(..)
    I = Integral(x in ClosedInterval(-Inf, Inf))
    eqs = I(u(x)) ~ 0
    domains = [x ∈ Interval(1.0, 2.0)]
    bcs = [u(1) ~ u(1)]
    chain = FastChain((x,p) -> x .* exp.(-x .^2))
    chain([1],Float64[])

    discretization = NeuralPDE.PhysicsInformedNN(chain,strategy_;init_params = Float64[])
    @named pde_system = PDESystem(eqs, bcs, domains, [x], [u(x)])
    sym_prob = SciMLBase.symbolic_discretize(pde_system, discretization)
    prob = SciMLBase.discretize(pde_system, discretization)
    inner_loss =prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents[1].loss_function
    exact_u = 0
    @test  inner_loss(ones(1,1), Float64[])[1] ≈ exact_u  rtol = 1e-9
end
