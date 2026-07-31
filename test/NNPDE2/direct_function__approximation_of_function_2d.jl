using NeuralPDE
using Test

@testset "Approximation of function 2D" begin
    using Optimization, OptimizationOptimisers, Random, DomainSets, Lux, Optimisers
    import DomainSets: Interval, infimum, supremum
    import OptimizationOptimJL: BFGS

    Random.seed!(110)

    @parameters x, y
    @variables u(..)
    func(x, y) = -cos(x) * cos(y) * exp(-((x - pi)^2 + (y - pi)^2))

    eq = [u(x, y) ~ func(x, y)]
    bc = [u(0, 0) ~ u(0, 0)]

    x0 = -10
    x_end = 10
    y0 = -10
    y_end = 10
    d = 0.4
    domain = [x ∈ Interval(x0, x_end), y ∈ Interval(y0, y_end)]
    hidden = 25
    chain = Chain(
        Dense(2, hidden, tanh), Dense(hidden, hidden, tanh),
        Dense(hidden, hidden, tanh), Dense(hidden, 1)
    )
    strategy = GridTraining(d)
    discretization = PhysicsInformedNN(chain, strategy)
    @named pde_system = PDESystem(eq, bc, domain, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    symprob = NeuralPDE.symbolic_discretize(pde_system, discretization)
    symprob.loss_functions.full_loss_function(symprob.flat_init_params, nothing)
    res = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(), maxiters = 1000)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, BFGS(), maxiters = 500)
    phi = discretization.phi
    xs = collect(x0:0.1:x_end)
    ys = collect(y0:0.1:y_end)
    u_predict = reshape(
        [first(phi([x, y], res.u)) for x in xs for y in ys],
        (length(xs), length(ys))
    )
    u_real = reshape([func(x, y) for x in xs for y in ys], (length(xs), length(ys)))
    diff_u = abs.(u_predict .- u_real)
    @test u_predict ≈ u_real rtol = 0.05
end
