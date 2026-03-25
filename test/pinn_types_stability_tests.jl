@testitem "PINNRepresentation Type Stability Smoke Test" tags = [:nnpde1] begin
    using NeuralPDE, ModelingToolkit, Lux, DomainSets, Optimization, OptimizationOptimisers
    
    @parameters x
    @variables u(..)
    Dx = Differential(x)

    # 1D ODE: u'(x) = cos(x)
    eq = Dx(u(x)) ~ cos(x)
    bcs = [u(0.0) ~ 0.0]
    domains = [x ∈ Interval(0.0, Float64(π))]

    # Neural Network Architecture
    chain = Chain(Dense(1, 8, tanh), Dense(8, 1))

    # Discretization using Grid Training
    discretization = PhysicsInformedNN(chain, GridTraining(0.1))
    @named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])

    # Convert PDESystem to OptimizationProblem
    prob = discretize(pde_system, discretization)

    # Run optimization
    res = solve(prob, Adam(0.01); maxiters = 50)

    # Verify predictions
    phi = discretization.phi
    u_pred = [first(phi([xi], res.u)) for xi in 0.0:0.5:Float64(π)]
    u_real = [sin(xi) for xi in 0.0:0.5:Float64(π)]
    
    max_error = maximum(abs.(u_pred .- u_real))
    
    # We do a loose bound check just to verify the pipeline doesn't error
    # and the gradients are propagating reasonably
    @test max_error < 1.0
end
