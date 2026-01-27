@testsetup module GPUNonlinearTestSetup
using NeuralPDE
using Symbolics: expand_derivatives
using Lux, Optimization, OptimizationOptimisers
using Random, ComponentArrays, LuxCUDA, CUDA
using NeuralPDE.GPUUtils

function callback(p, l)
    if p.iter == 1 || p.iter % 100 == 0
        println("GPU Nonlinear Test - Loss: $l after $(p.iter) iterations")
    end
    return false
end

const gpud = CUDA.functional() ? gpu_device() : nothing
export gpud, callback, transform_power_ops, should_apply_gpu_transform
end

@testitem "Symbolic Power Transformation" tags = [:gpu_nonlinear] setup = [GPUNonlinearTestSetup] begin
    using Symbolics
    
    @variables x u(..)
    Dx = Differential(x)
    
    # Test basic transformation: u^2 → u * u
    expr2 = u(x)^2
    transformed2 = transform_power_ops(expr2)
    @test Symbolics.simplify(transformed2 - u(x)*u(x)) == 0    

    # Test: u^3 → u * u * u
    expr3 = u(x)^3
    transformed3 = transform_power_ops(expr3)
    @test Symbolics.simplify(transformed3 - u(x)*u(x)*u(x)) == 0

    # Test derivative compatibility: symbolic differentiation should work after transformation
    expr_deriv = Dx(u(x)^3)
    transformed_deriv = transform_power_ops(expr_deriv)
    expanded = expand_derivatives(transformed_deriv)
    
    # Should not crash and should produce a valid expression
    @test !isnothing(expanded)
    @test expanded isa Union{Num, SymbolicUtils.Term}
    
    # Test non-integer exponents: should not be transformed
    expr_nonint = u(x)^2.5
    transformed_nonint = transform_power_ops(expr_nonint)
    @test Symbolics.simplify(transformed_nonint - u(x)^2.5) == 0
    
    # Test edge cases: u^0 = 1, u^1 = u
    @test transform_power_ops(u(x)^1) == u(x)
    @test transform_power_ops(u(x)^0) == 1
end

@testitem "GPU Device Detection" tags = [:gpu_nonlinear] setup = [GPUNonlinearTestSetup] begin
    using ComponentArrays
    
    # Test with nothing: should return false
    @test should_apply_gpu_transform(nothing) == false
    
    # Test with CPU parameters: should return false  
    cpu_params = ComponentArray(a = [1.0, 2.0, 3.0])
    @test should_apply_gpu_transform(cpu_params) == false
    
    # Test with GPU parameters (if CUDA available)
    if CUDA.functional()
        gpu_params = ComponentArray(a = [1.0, 2.0, 3.0]) |> gpud
        @test should_apply_gpu_transform(gpu_params) == true
    end
end

@testitem "Nonlinear PDE u^2 - CUDA" tags = [:cuda, :gpu_nonlinear] setup = [GPUNonlinearTestSetup] begin
    using CUDA
    import DomainSets: Interval
    
    CUDA.functional() || return  # Skip if CUDA not available
    
    Random.seed!(100)
    
    @parameters x
    @variables u(..)
    Dx = Differential(x)
    
    # Simple nonlinear PDE: u^2 = 0 with boundary condition u(0) = 0
    # This tests the symbolic transformation of power operations in PDE equations
    eq = u(x)^2 ~ 0.0
    bcs = [u(0.0) ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.0)]
    
    # Neural network: small configuration for unit testing
    inner = 10
    chain = Chain(Dense(1, inner, tanh), Dense(inner, inner, tanh), Dense(inner, 1))
    
    strategy = GridTraining(0.1)
    ps = Lux.initialparameters(Random.default_rng(), chain) |> ComponentArray |> gpud |> f64
    
    discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
    
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    
    # Solve: power transformation should enable differentiation
    res = solve(prob, Adam(0.01); maxiters = 200, callback = callback)
    
    # Verify solution integrity: no NaN or Inf values
    @test !any(isnan, res.u)
    @test all(isfinite, res.u)
end

@testitem "Nonlinear PDE Dx(u^3) - CUDA" tags = [:cuda, :gpu_nonlinear] setup = [GPUNonlinearTestSetup] begin
    using CUDA
    import DomainSets: Interval
    
    CUDA.functional() || return  # Skip if CUDA not available
    
    Random.seed!(200)
    
    @parameters x
    @variables u(..)
    Dx = Differential(x)
    
    # Test case from issue #914: Dx(u^3)
    # This case produced NaN in automatic differentiation prior to power operation expansion
    # The fix transforms u^3 → u * u * u, enabling chain rule application
    eq = Dx(u(x)^3) ~ 0.0
    bcs = [u(0.0) ~ 0.0]
    domains = [x ∈ Interval(0.0, 1.0)]
    
    # Neural network: small configuration for unit testing
    inner = 10
    chain = Chain(Dense(1, inner, tanh), Dense(inner, inner, tanh), Dense(inner, 1))
    
    strategy = QuasiRandomTraining(1000)
    ps = Lux.initialparameters(Random.default_rng(), chain) |> ComponentArray |> gpud |> f64
    
    discretization = PhysicsInformedNN(chain, strategy; init_params = ps)
    
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    
    # Solve: this was the case that generated NaN before the fix
    res = solve(prob, Adam(0.01); maxiters = 200, callback = callback)
    
    # Verify solution: the main assertion that the fix works
    @test !any(isnan, res.u)
    @test all(isfinite, res.u)
end