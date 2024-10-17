@testsetup module NeuralAdapterTestSetup

function callback(p, l)
    (p.iter == 1 || p.iter % 500 == 0) &&
        println("Current loss is: $l after $(p.iter) iterations")
    return false
end

export callback

end

@testitem "Neural Adapter: 2D Poisson" tags=[:neuraladapter] setup=[NeuralAdapterTestSetup] begin
    using Optimization, Lux, OptimizationOptimisers, Statistics, ComponentArrays, Random,
          LinearAlgebra
    import ModelingToolkit: Interval, infimum, supremum

    Random.seed!(100)

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)

    # Initial and boundary conditions
    bcs = [
        u(0, y) ~ 0.0,
        u(1, y) ~ -sinpi(1) * sinpi(y),
        u(x, 0) ~ 0.0,
        u(x, 1) ~ -sinpi(x) * sinpi(1)
    ]
    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
    quadrature_strategy = QuadratureTraining(
        reltol = 1e-3, abstol = 1e-6, maxiters = 50, batch = 100)
    chain1 = Chain(Dense(2, 8, tanh), Dense(8, 8, tanh), Dense(8, 1))
    discretization = PhysicsInformedNN(chain1, quadrature_strategy)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
    prob = discretize(pde_system, discretization)
    res = solve(prob, Adam(5e-3); callback, maxiters = 2000)
    phi = discretization.phi

    xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x, y) = (sinpi(x) * sinpi(y)) / (2pi^2)

    u_predict = [first(phi([x, y], res.u)) for x in xs for y in ys]
    u_real = [analytic_sol_func(x, y) for x in xs for y in ys]

    @test u_predict≈u_real atol=5e-2 norm=Base.Fix2(norm, Inf)

    chain2 = Chain(Dense(2, 8, tanh), Dense(8, 8, tanh), Dense(8, 1))
    initp, st = Lux.setup(Random.default_rng(), chain2)
    init_params2 = ComponentArray{Float64}(initp)

    loss(cord, θ) = first(chain2(cord, θ, st)) .- phi(cord, res.u)

    grid_strategy = GridTraining(0.05)
    quadrature_strategy = QuadratureTraining(
        reltol = 1e-3, abstol = 1e-6, maxiters = 50, batch = 100)
    stochastic_strategy = StochasticTraining(1000)
    quasirandom_strategy = QuasiRandomTraining(1000, minibatch = 200, resampling = true)

    @testset "$(nameof(typeof(strategy_)))" for strategy_ in [
        grid_strategy, quadrature_strategy, stochastic_strategy, quasirandom_strategy]
        prob_ = neural_adapter(loss, init_params2, pde_system, strategy_)
        res_ = solve(prob_, Optimisers.Adam(5e-3); callback, maxiters = 2000)
        discretization = PhysicsInformedNN(chain2, strategy_; init_params = res_.u)
        phi_ = discretization.phi

        u_predict_ = [first(phi_([x, y], res_.u)) for x in xs for y in ys]
        @test u_predict_≈u_real atol=5e-2 norm=Base.Fix2(norm, Inf)
    end
end

@testitem "Neural Adapter: 2D Poisson, domain decomposition" tags=[:neuraladapter] setup=[NeuralAdapterTestSetup] begin
    using Optimization, Lux, OptimizationOptimisers, Statistics, ComponentArrays, Random,
          LinearAlgebra
    import ModelingToolkit: Interval, infimum, supremum

    Random.seed!(100)

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)

    bcs = [
        u(0, y) ~ 0.0,
        u(1, y) ~ -sinpi(1) * sinpi(y),
        u(x, 0) ~ 0.0,
        u(x, 1) ~ -sinpi(x) * sinpi(1)
    ]

    # Space
    x_0, x_end = 0.0, 1.0
    x_domain = Interval(x_0, x_end)
    y_domain = Interval(0.0, 1.0)
    domains = [x ∈ x_domain, y ∈ y_domain]
    count_decomp = 10

    # Neural network
    chains = [Chain(Dense(2, 8, tanh), Dense(8, 8, tanh), Dense(8, 1))
              for _ in 1:count_decomp]

    xs_ = infimum(x_domain):(1 / count_decomp):supremum(x_domain)
    xs_domain = [(xs_[i], xs_[i + 1]) for i in 1:(length(xs_) - 1)]
    domains_map = map(xs_domain) do (xs_dom)
        x_domain_ = Interval(xs_dom...)
        domains_ = [x ∈ x_domain_, y ∈ y_domain]
    end

    analytic_sol_func(x, y) = (sinpi(x) * sinpi(y)) / (2pi^2)
    function create_bcs(x_domain_, phi_bound)
        x_0, x_e = x_domain_.left, x_domain_.right
        if x_0 == 0.0
            bcs = [u(0, y) ~ 0.0, u(x_e, y) ~ analytic_sol_func(x_e, y),
                u(x, 0) ~ 0.0, u(x, 1) ~ -sinpi(x) * sinpi(1)]
            return bcs
        end
        bcs = [u(x_0, y) ~ phi_bound(x_0, y), u(x_e, y) ~ analytic_sol_func(x_e, y),
            u(x, 0) ~ 0.0, u(x, 1) ~ -sinpi(x) * sinpi(1)]
        bcs
    end

    reses = []
    phis = []
    pde_system_map = []

    for i in 1:count_decomp
        domains_ = domains_map[i]
        phi_in(cord) = phis[i - 1](cord, reses[i - 1].u)
        phi_bound(x, y) = phi_in(vcat(x, y))
        @register_symbolic phi_bound(x, y)
        Base.Broadcast.broadcasted(::typeof(phi_bound), x, y) = phi_bound(x, y)
        bcs_ = create_bcs(domains_[1].domain, phi_bound)
        @named pde_system_ = PDESystem(eq, bcs_, domains_, [x, y], [u(x, y)])
        push!(pde_system_map, pde_system_)

        strategy = GridTraining([0.1 / count_decomp, 0.1])
        discretization = PhysicsInformedNN(chains[i], strategy)
        prob = discretize(pde_system_, discretization)
        res_ = solve(prob, Optimisers.Adam(5e-3); callback, maxiters = 2000)
        phi = discretization.phi

        push!(reses, res_)
        push!(phis, phi)
    end

    function compose_result(dx)
        u_predict_array = Float64[]
        diff_u_array = Float64[]
        ys = infimum(domains[2].domain):dx:supremum(domains[2].domain)
        xs_ = infimum(x_domain):dx:supremum(x_domain)
        xs = collect(xs_)
        function index_of_interval(x_)
            for (i, x_domain) in enumerate(xs_domain)
                if x_ <= x_domain[2] && x_ >= x_domain[1]
                    return i
                end
            end
        end
        for x_ in xs
            i = index_of_interval(x_)
            u_predict_sub = [first(phis[i]([x_, y], reses[i].u)) for y in ys]
            u_real_sub = [analytic_sol_func(x_, y) for y in ys]
            diff_u_sub = u_predict_sub .- u_real_sub
            append!(u_predict_array, u_predict_sub)
            append!(diff_u_array, diff_u_sub)
        end
        xs, ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
        u_predict = reshape(u_predict_array, (length(xs), length(ys)))
        diff_u = reshape(diff_u_array, (length(xs), length(ys)))
        u_predict, diff_u
    end

    dx = 0.01
    u_predict, diff_u = compose_result(dx)

    chain2 = Chain(Dense(2, 18, tanh), Dense(18, 18, tanh), Dense(18, 18, tanh),
        Dense(18, 18, tanh), Dense(18, 1))

    initp, st = Lux.setup(Random.default_rng(), chain2)
    init_params2 = ComponentArray{Float64}(initp)

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

    losses = map(1:count_decomp) do i
        loss(cord, θ) = first(chain2(cord, θ, st)) .- phis[i](cord, reses[i].u)
    end

    prob_ = neural_adapter(
        losses, init_params2, pde_system_map, GridTraining([0.1 / count_decomp, 0.1]))
    res_ = solve(prob_, Adam(5e-3); callback, maxiters = 2000)

    prob_ = neural_adapter(losses, res_.u, pde_system_map, GridTraining(0.01))
    res_ = solve(prob_, Adam(5e-3); callback, maxiters = 2000)

    phi_ = NeuralPDE.Phi(chain2)
    xs, ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
    u_predict_ = reshape(
        [first(phi_([x, y], res_.u)) for x in xs for y in ys], (length(xs), length(ys)))
    u_real = reshape(
        [analytic_sol_func(x, y) for x in xs for y in ys], (length(xs), length(ys)))
    diff_u_ = u_predict_ .- u_real

    @test u_predict≈u_real atol=5e-2 norm=Base.Fix2(norm, Inf)
    @test u_predict_≈u_real atol=5e-2 norm=Base.Fix2(norm, Inf)
end
