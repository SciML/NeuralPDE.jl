using NeuralPDE, Test, Optimization, OptimizationOptimJL, OptimizationOptimisers,
      QuasiMonteCarlo, Random, DomainSets, Integrals, Cubature, OrdinaryDiffEq,
      ComponentArrays, Lux
import ModelingToolkit: Interval, infimum, supremum

@testset "Fokker-Planck" begin
    # the example took from this article https://arxiv.org/abs/1910.10503
    @parameters x
    @variables p(..)
    Dx = Differential(x)
    Dxx = Differential(x)^2
    α = 0.3
    β = 0.5
    _σ = 0.5
    # Discretization
    dx = 0.01
    # here we use normalization condition: dx*p(x) ~ 1, in order to get non-zero solution.
    # (α - 3*β*x^2)*p(x) + (α*x - β*x^3)*Dx(p(x)) ~ (_σ^2/2)*Dxx(p(x))
    eq = [Dx((α * x - β * x^3) * p(x)) ~ (_σ^2 / 2) * Dxx(p(x))]
    x_0 = -2.2
    x_end = 2.2
    # Initial and boundary conditions
    bcs = [p(x_0) ~ 0.0, p(x_end) ~ 0.0]

    # Space and time domains
    domains = [x ∈ Interval(-2.2, 2.2)]

    # Neural network
    inn = 18
    chain = Chain(Dense(1, inn, σ), Dense(inn, inn, σ), Dense(inn, inn, σ), Dense(inn, 1))
    init_params = ComponentArray{Float64}(Lux.initialparameters(
        Random.default_rng(), chain))
    lb = [x_0]
    ub = [x_end]
    function norm_loss_function(phi, θ, p)
        function inner_f(x, θ)
            dx * phi(x, θ) .- 1
        end
        prob1 = IntegralProblem(inner_f, (lb, ub), θ)
        norm2 = solve(prob1, HCubatureJL(), reltol = 1e-8, abstol = 1e-8, maxiters = 10)
        return abs(norm2[1])
    end
    discretization = PhysicsInformedNN(chain, GridTraining(dx); init_params = init_params,
        additional_loss = norm_loss_function)
    @named pde_system = PDESystem(eq, bcs, domains, [x], [p(x)])
    prob = discretize(pde_system, discretization)
    sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions
    phi = discretization.phi
    cb_ = function (p, l)
        println("loss: ", l)
        println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
        println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
        println("additional_loss: ", norm_loss_function(phi, p.u, nothing))
        return false
    end
    res = solve(prob, OptimizationOptimJL.LBFGS(), maxiters = 400, callback = cb_)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, OptimizationOptimJL.BFGS(), maxiters = 2000, callback = cb_)
    C = 142.88418699042
    analytic_sol_func(x) = C * exp((1 / (2 * _σ^2)) * (2 * α * x^2 - β * x^4))
    xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(x) for x in xs]
    u_predict = [first(phi(x, res.u)) for x in xs]
    @test u_predict≈u_real rtol=1e-3

    ### No init_params
    discretization = PhysicsInformedNN(
        chain, GridTraining(dx); additional_loss = norm_loss_function)
    @named pde_system = PDESystem(eq, bcs, domains, [x], [p(x)])
    prob = discretize(pde_system, discretization)
    sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions
    phi = discretization.phi
    cb_ = function (p, l)
        println("loss: ", l)
        println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
        println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
        println("additional_loss: ", norm_loss_function(phi, p.u, nothing))
        return false
    end
    res = solve(prob, OptimizationOptimJL.LBFGS(), maxiters = 400, callback = cb_)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, OptimizationOptimJL.BFGS(), maxiters = 2000, callback = cb_)
    C = 142.88418699042
    analytic_sol_func(x) = C * exp((1 / (2 * _σ^2)) * (2 * α * x^2 - β * x^4))
    xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]
    u_real = [analytic_sol_func(x) for x in xs]
    u_predict = [first(phi(x, res.u)) for x in xs]
    @test u_predict≈u_real rtol=1e-3
end

@testset "Lorenz System" begin
    @parameters t, σ_, β, ρ
    @variables x(..), y(..), z(..)
    Dt = Differential(t)
    eqs = [Dt(x(t)) ~ σ_ * (y(t) - x(t)),
        Dt(y(t)) ~ x(t) * (ρ - z(t)) - y(t),
        Dt(z(t)) ~ x(t) * y(t) - β * z(t)]

    bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 1.0)]
    dt = 0.05

    input_ = length(domains)
    n = 12
    chain = [Chain(Dense(input_, n, tanh), Dense(n, n, σ), Dense(n, 1)) for _ in 1:3]
    #Generate Data
    function lorenz!(du, u, p, t)
        du[1] = 10.0 * (u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
    end

    u0 = [1.0; 0.0; 0.0]
    tspan = (0.0, 1.0)
    prob = ODEProblem(lorenz!, u0, tspan)
    sol = solve(prob, Tsit5(), dt = 0.1)
    ts = [infimum(d.domain):dt:supremum(d.domain) for d in domains][1]

    function getData(sol)
        data = []
        us = hcat(sol(ts).u...)
        ts_ = hcat(sol(ts).t...)
        return [us, ts_]
    end

    data = getData(sol)

    #Additional Loss Function
    init_params = [Float64.(ComponentArray(Lux.setup(Random.default_rng(), chain[i])[1]))
                   for i in 1:3]
    names = (:x, :y, :z)
    flat_init_params = ComponentArray(NamedTuple{names}(i for i in init_params))

    acum = [0; accumulate(+, length.(init_params))]
    sep = [(acum[i] + 1):acum[i + 1] for i in 1:(length(acum) - 1)]
    (u_, t_) = data
    len = length(data[2])

    function additional_loss(phi, θ, p)
        return sum(sum(abs2, phi[i](t_, getproperty(θ, names[i])) .- u_[[i], :]) /
                   len
        for i in 1:1:3)
    end

    discretization = PhysicsInformedNN(chain, GridTraining(dt);
        init_params = flat_init_params, param_estim = true, additional_loss)

    additional_loss(discretization.phi, flat_init_params, nothing)
    @named pde_system = PDESystem(eqs, bcs, domains,
        [t], [x(t), y(t), z(t)], [σ_, ρ, β],
        defaults = Dict([p => 1.0 for p in [σ_, ρ, β]]))
    prob = discretize(pde_system, discretization)
    sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
    sym_prob.loss_functions.full_loss_function(
        ComponentArray(depvar = flat_init_params, p = ones(3)), Float64[])

    res = solve(prob, OptimizationOptimJL.BFGS(); maxiters = 6000)
    p_ = res.u[(end - 2):end]
    @test sum(abs2, p_[1] - 10.00) < 0.1
    @test sum(abs2, p_[2] - 28.00) < 0.1
    @test sum(abs2, p_[3] - (8 / 3)) < 0.1

    ### No init_params
    discretization = PhysicsInformedNN(
        chain, GridTraining(dt); param_estim = true, additional_loss)

    additional_loss(discretization.phi, flat_init_params, nothing)
    @named pde_system = PDESystem(eqs, bcs, domains,
        [t], [x(t), y(t), z(t)], [σ_, ρ, β],
        defaults = Dict([p => 1.0 for p in [σ_, ρ, β]]))
    prob = discretize(pde_system, discretization)
    sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
    sym_prob.loss_functions.full_loss_function(sym_prob.flat_init_params, nothing)
    res = solve(prob, OptimizationOptimJL.BFGS(); maxiters = 6000)
    p_ = res.u[(end - 2):end]
    @test sum(abs2, p_[1] - 10.00) < 0.1
    @test sum(abs2, p_[2] - 28.00) < 0.1
    @test sum(abs2, p_[3] - (8 / 3)) < 0.1
end

@testset "Approximation from data and additional_loss" begin
    @parameters x
    @variables u(..)
    eq = [u(0) ~ u(0)]
    bc = [u(0) ~ u(0)]
    x0 = 0
    x_end = pi
    dx = pi / 10
    domain = [x ∈ Interval(x0, x_end)]
    hidden = 10
    chain = Chain(Dense(1, hidden, tanh), Dense(hidden, hidden, sin),
        Dense(hidden, hidden, tanh), Dense(hidden, 1))
    strategy = GridTraining(dx)
    xs = collect(x0:dx:x_end)'
    aproxf_(x) = @. cos(pi * x)
    data = aproxf_(xs)
    function additional_loss_(phi, θ, p)
        sum(abs2, phi(xs, θ) .- data)
    end
    discretization = PhysicsInformedNN(chain, strategy; additional_loss = additional_loss_)
    @named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
    prob = discretize(pde_system, discretization)
    sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
    flat_init_params = sym_prob.flat_init_params
    phi = discretization.phi
    phi(xs, flat_init_params)
    additional_loss_(phi, flat_init_params, nothing)
    res = solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
    prob = remake(prob, u0 = res.u)
    res = solve(prob, OptimizationOptimJL.BFGS(), maxiters = 500)
    @test phi(xs, res.u)≈aproxf_(xs) rtol=0.01
end
