using Test 
using .NeuralPDE
using Lux
using Random
using ComponentArrays
using ModelingToolkit
using OptimizationProblems
import ModelingToolkit: Interval


@testset "PIPN Tests" begin
    @testset "PIPN Construction" begin
        chain = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 1))
        pipn = PIPN(chain)
        @test pipn isa PIPN
        @test pipn.shared_mlp1 isa Lux.Chain
        @test pipn.shared_mlp2 isa Lux.Chain
        @test pipn.after_pool_mlp isa Lux.Chain
        @test pipn.final_layer isa Lux.Dense
    end

    @testset "PIPN Forward Pass" begin
        chain = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 1))
        pipn = PIPN(chain)
        x = rand(Float32, 2, 100)
        println("Test input size: ", size(x))
        ps, st = init_pipn_params(pipn)
        y, _ = pipn(x, ps, st)
        @test size(y) == (1, 100)
    end

    @testset "PIPN Parameter Initialization" begin
        chain = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 1))
        pipn = PIPN(chain)
        ps, st = init_pipn_params(pipn)
        @test ps isa NamedTuple
        @test st isa NamedTuple
    end

    @testset "PIPN Parameter Conversion" begin
        chain = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 1))
        pipn = PIPN(chain)
        ps, _ = init_pipn_params(pipn)
        flat_ps = ComponentArray(ps)
        converted_ps = vector_to_parameters(flat_ps, pipn)
        @test converted_ps isa ComponentArray
    end

    @testset "PIPN with PDESystem" begin
        @parameters x t
        @variables u(..)
        Dt = Differential(t)
        Dxx = Differential(x)^2
        eq = Dt(u(x,t)) ~ Dxx(u(x,t))

        # Define domain
        x_min = 0.0
        x_max = 1.0
        t_min = 0.0
        t_max = 1.0

        # Use DomainSets for domain definition
        domains = [x ∈ Interval(x_min, x_max),
                   t ∈ Interval(t_min, t_max)]

        bcs = [u(x,0) ~ sin(π*x),
               u(0,t) ~ 0.0,
               u(1,t) ~ 0.0]

        @named pde_system = PDESystem(eq, bcs, domains, [x,t], [u(x,t)])

        chain = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 1))
        strategy = GridTraining(0.1)
        discretization = PhysicsInformedNN(chain, strategy)
        
        prob = discretize(pde_system, discretization)
        
        @test prob isa OptimizationProblem
    end
end