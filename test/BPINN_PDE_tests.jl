@testitem "BPINN PDE I: 1D Periodic System" tags = [:pdebpinn] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters t
    @variables u(..)
    Dt = Differential(t)
    eq = Dt(u(t)) - cospi(2t) ~ 0
    bcs = [u(0.0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]

    chainl = Chain(Dense(1, 6, tanh), Dense(6, 1))
    initl, st = Lux.setup(Random.default_rng(), chainl)
    @named pde_system = PDESystem(eq, bcs, domains, [t], [u(t)])

    # non adaptive case
    discretization = BayesianPINN([chainl], GridTraining([0.01]))

    sol1 = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 1500, bcstd = [0.01],
        phystd = [0.01], priorsNNw = (0.0, 1.0), saveats = [1 / 50.0]
    )

    analytic_sol_func(u0, t) = u0 + sinpi(2t) / (2pi)
    ts = vec(sol1.timepoints[1])
    u_real = [analytic_sol_func(0.0, t) for t in ts]
    u_predict = pmean(sol1.ensemblesol[1])

    # absol tests
    @test mean(abs, u_predict .- u_real) < 8.0e-2
end

@testitem "BPINN PDE II: 1D ODE" tags = [:pdebpinn] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)

    # 1D ODE
    eq = Dθ(u(θ)) ~
        θ^3 + 2.0f0 * θ + (θ^2) * ((1.0f0 + 3 * (θ^2)) / (1.0f0 + θ + (θ^3))) -
        u(θ) * (θ + ((1.0f0 + 3.0f0 * (θ^2)) / (1.0f0 + θ + θ^3)))

    # Initial and boundary conditions
    bcs = [u(0.0) ~ 1.0f0]

    # Space and time domains
    domains = [θ ∈ Interval(0.0f0, 1.0f0)]

    # Discretization
    dt = 0.1f0

    # Neural network
    chain = Chain(Dense(1, 12, σ), Dense(12, 1))

    discretization = BayesianPINN([chain], GridTraining([0.01]))

    @named pde_system = PDESystem(eq, bcs, domains, [θ], [u])

    sol1 = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 500, bcstd = [0.1],
        phystd = [0.05], priorsNNw = (0.0, 10.0), saveats = [1 / 100.0]
    )

    analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
    ts = sol1.timepoints[1]
    u_real = vec([analytic_sol_func(t) for t in ts])
    u_predict = pmean(sol1.ensemblesol[1])
    @test u_predict ≈ u_real atol = 0.8
end

@testitem "BPINN PDE III: 3rd Degree ODE" tags = [:pdebpinn] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters x
    @variables u(..), Dxu(..), Dxxu(..), O1(..), O2(..)
    Dxxx = Differential(x)^3
    Dx = Differential(x)

    # ODE
    eq = Dx(Dxxu(x)) ~ cospi(x)

    # Initial and boundary conditions
    ep = (cbrt(eps(eltype(Float64))))^2 / 6

    bcs = [
        u(0.0) ~ 0.0,
        u(1.0) ~ cospi(1.0),
        Dxu(1.0) ~ 1.0,
        Dxu(x) ~ Dx(u(x)) + ep * O1(x),
        Dxxu(x) ~ Dx(Dxu(x)) + ep * O2(x),
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0)]

    # Neural network
    chain = [
        Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1)),
        Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1)),
        Chain(Dense(1, 10, tanh), Dense(10, 10, tanh), Dense(10, 1)),
        Chain(Dense(1, 4, tanh), Dense(4, 1)),
        Chain(Dense(1, 4, tanh), Dense(4, 1)),
    ]

    discretization = BayesianPINN(chain, GridTraining(0.01))

    @named pde_system = PDESystem(
        eq, bcs, domains, [x],
        [u(x), Dxu(x), Dxxu(x), O1(x), O2(x)]
    )

    sol1 = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 200,
        bcstd = [0.01, 0.01, 0.01, 0.01, 0.01], phystd = [0.005],
        priorsNNw = (0.0, 10.0), saveats = [1 / 100.0]
    )

    analytic_sol_func(x) = (π * x * (-x + (π^2) * (2 * x - 3) + 1) - sinpi(x)) / (π^3)

    u_predict = pmean(sol1.ensemblesol[1])
    xs = vec(sol1.timepoints[1])
    u_real = [analytic_sol_func(x) for x in xs]
    @test u_predict ≈ u_real atol = 0.5
end

@testitem "BPINN PDE IV: 2D Poisson" tags = [:pdebpinn] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters x y
    @variables u(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2

    # 2D PDE
    eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

    # Boundary conditions
    bcs = [
        u(0, y) ~ 0.0,
        u(1, y) ~ 0.0,
        u(x, 0) ~ 0.0,
        u(x, 1) ~ 0.0,
    ]

    # Space and time domains
    domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]

    # Discretization
    dt = 0.1f0

    # Neural network
    chain = Chain(Dense(2, 9, σ), Dense(9, 9, σ), Dense(9, 1))

    dx = 0.04
    discretization = BayesianPINN([chain], GridTraining(dx))

    @named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

    sol = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 200,
        bcstd = [0.003, 0.003, 0.003, 0.003], phystd = [0.003],
        priorsNNw = (0.0, 10.0), saveats = [1 / 100.0, 1 / 100.0]
    )

    xs = sol.timepoints[1]
    analytic_sol_func(x, y) = (sinpi(x) * sinpi(y)) / (2pi^2)

    u_predict = pmean(sol.ensemblesol[1])
    u_real = [analytic_sol_func(xs[:, i][1], xs[:, i][2]) for i in 1:length(xs[1, :])]
    @test u_predict ≈ u_real rtol = 0.5
end

@testitem "BPINN PDE: Translating from Flux" tags = [:pdebpinn] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum
    import Flux

    Random.seed!(100)

    @parameters θ
    @variables u(..)
    Dθ = Differential(θ)

    # 1D ODE
    eq = Dθ(u(θ)) ~
        θ^3 + 2.0f0 * θ + (θ^2) * ((1.0f0 + 3 * (θ^2)) / (1.0f0 + θ + (θ^3))) -
        u(θ) * (θ + ((1.0f0 + 3.0f0 * (θ^2)) / (1.0f0 + θ + θ^3)))

    # Initial and boundary conditions
    bcs = [u(0.0) ~ 1.0f0]

    # Space and time domains
    domains = [θ ∈ Interval(0.0f0, 1.0f0)]

    # Neural network
    chain = Flux.Chain(Flux.Dense(1, 12, Flux.σ), Flux.Dense(12, 1))

    discretization = BayesianPINN([chain], GridTraining([0.01]))
    @test discretization.chain[1] isa Lux.AbstractLuxLayer

    @named pde_system = PDESystem(eq, bcs, domains, [θ], [u])

    sol = ahmc_bayesian_pinn_pde(
        pde_system, discretization; draw_samples = 500,
        bcstd = [0.1], phystd = [0.05], priorsNNw = (0.0, 10.0), saveats = [1 / 100.0]
    )

    analytic_sol_func(t) = exp(-(t^2) / 2) / (1 + t + t^3) + t^2
    ts = sol.timepoints[1]
    u_real = vec([analytic_sol_func(t) for t in ts])
    u_predict = pmean(sol.ensemblesol[1])

    @test u_predict ≈ u_real atol = 0.8
end

@testitem "BPINN PDE Inv I: 1D Periodic System" tags = [:pdebpinn] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters t p
    @variables u(..)

    Dt = Differential(t)
    eqs = Dt(u(t)) - cos(p * t) ~ 0
    bcs = [u(0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 2.0)]

    chainl = Lux.Chain(Lux.Dense(1, 6, tanh), Lux.Dense(6, 6, tanh), Lux.Dense(6, 1))
    initl, st = Lux.setup(Random.default_rng(), chainl)

    @named pde_system = PDESystem(
        eqs,
        bcs,
        domains,
        [t],
        [u(t)],
        [p],
        defaults = Dict([p => 4.0])
    )

    analytic_sol_func1(u0, t) = u0 + sinpi(2t) / (2π)
    timepoints = collect(0.0:(1 / 100.0):2.0)
    u = [analytic_sol_func1(0.0, timepoint) for timepoint in timepoints]
    u = u .+ (u .* 0.2) .* randn(size(u))
    dataset = [hcat(u, timepoints)]

    # BPINNs are formulated with a mesh that must stay the same throughout sampling (as of now)
    @testset "$(nameof(typeof(strategy)))" for strategy in [
            # StochasticTraining(200),
            # QuasiRandomTraining(200),
            GridTraining([0.02]),
        ]
        discretization = BayesianPINN(
            [chainl], strategy; param_estim = true,
            dataset = [dataset, nothing]
        )

        sol1 = ahmc_bayesian_pinn_pde(
            pde_system,
            discretization;
            draw_samples = 1500,
            bcstd = [0.02],
            phystd = [0.02], l2std = [0.02],
            priorsNNw = (0.0, 1.0),
            saveats = [1 / 50.0],
            param = [LogNormal(6.0, 0.5)]
        )

        param = 2 * π
        ts = vec(sol1.timepoints[1])
        u_real = [analytic_sol_func1(0.0, t) for t in ts]
        u_predict = pmean(sol1.ensemblesol[1])

        @test mean(abs, u_predict .- u_real) < 8.0e-2
        @test sol1.estimated_de_params[1] ≈ param rtol = 0.1
    end
end

@testitem "BPINN PDE Inv II: Lorenz System" tags = [:pdebpinn] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    @parameters t, σ_
    @variables x(..), y(..), z(..)
    Dt = Differential(t)
    eqs = [
        Dt(x(t)) ~ σ_ * (y(t) - x(t)),
        Dt(y(t)) ~ x(t) * (28.0 - z(t)) - y(t),
        Dt(z(t)) ~ x(t) * y(t) - 8.0 / 3.0 * z(t),
    ]

    bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
    domains = [t ∈ Interval(0.0, 1.0)]

    input_ = length(domains)
    n = 7
    chain = [
        Chain(Dense(input_, n, tanh), Dense(n, n, tanh), Dense(n, 1)),
        Chain(Dense(input_, n, tanh), Dense(n, n, tanh), Dense(n, 1)),
        Chain(Dense(input_, n, tanh), Dense(n, n, tanh), Dense(n, 1)),
    ]

    # Generate Data
    function lorenz!(du, u, p, t)
        du[1] = 10.0 * (u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8.0 / 3.0) * u[3]
    end

    u0 = [1.0; 0.0; 0.0]
    tspan = (0.0, 1.0)
    prob = ODEProblem(lorenz!, u0, tspan)
    sol = solve(prob, Tsit5(), dt = 0.01, saveat = 0.05)
    ts = sol.t
    us = hcat(sol.u...)
    us = us .+ ((0.05 .* randn(size(us))) .* us)
    ts_ = hcat(ts...)[1, :]
    dataset = [hcat(us[i, :], ts_) for i in 1:3]

    discretization = BayesianPINN(
        chain, GridTraining([0.01]); param_estim = true,
        dataset = [dataset, nothing]
    )

    @named pde_system = PDESystem(
        eqs, bcs, domains,
        [t], [x(t), y(t), z(t)], [σ_], defaults = Dict([p => 1.0 for p in [σ_]])
    )

    sol1 = ahmc_bayesian_pinn_pde(
        pde_system,
        discretization;
        draw_samples = 50,
        bcstd = [0.3, 0.3, 0.3],
        phystd = [0.1, 0.1, 0.1],
        l2std = [1, 1, 1],
        priorsNNw = (0.0, 1.0),
        saveats = [0.01],
        param = [Normal(12.0, 2)]
    )

    idealp = 10.0
    p_ = sol1.estimated_de_params[1]
    @test sum(abs, pmean(p_) - 10.0) < 0.3 * idealp[1]
    # @test sum(abs, pmean(p_[2]) - (8 / 3)) < 0.3 * idealp[2]
end

@testitem "BPINN PDE Inv III: Improved Parametric Kuromo-Sivashinsky Equation solve" tags = [:pdebpinn] begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    function recur_expression(exp, Dict_differentials)
        for in_exp in exp.args
            if !(in_exp isa Expr)
                # skip +,== symbols, characters etc
                continue

            elseif in_exp.args[1] isa ModelingToolkit.Differential
                # first symbol of differential term
                # Dict_differentials for masking differential terms
                # and resubstituting differentials in equations after putting in interpolations
                # temp = in_exp.args[end]
                Dict_differentials[eval(in_exp)] = Symbolics.variable("diff_$(length(Dict_differentials) + 1)")
                return
            else
                recur_expression(in_exp, Dict_differentials)
            end
        end
    end

    @parameters α
    @variables x, t
    @syms u(x, t)
    Dt = Differential(t)
    Dx = Differential(x)
    Dx2 = Differential(x)^2
    Dx3 = Differential(x)^3
    Dx4 = Differential(x)^4

    # α = 1 (KS equation to be parametric in a)
    β = 4
    γ = 1
    eq = Dt(u(x, t)) + u(x, t) * Dx(u(x, t)) + α * Dx2(u(x, t)) + β * Dx3(u(x, t)) +
        γ * Dx4(u(x, t)) ~ 0

    u_analytic(x, t; z = -x / 2 + t) = 11 + 15 * tanh(z) - 15 * tanh(z)^2 - 15 * tanh(z)^3
    du(x, t; z = -x / 2 + t) = 15 / 2 * (tanh(z) + 1) * (3 * tanh(z) - 1) * sech(z)^2

    bcs = [
        u(x, 0) ~ u_analytic(x, 0),
        u(-10, t) ~ u_analytic(-10, t),
        u(10, t) ~ u_analytic(10, t),
        Dx(u(-10, t)) ~ du(-10, t),
        Dx(u(10, t)) ~ du(10, t),
    ]

    # Space and time domains
    domains = [
        x ∈ Interval(-10.0, 10.0),
        t ∈ Interval(0.0, 1.0),
    ]

    # Discretization
    dx = 0.4
    dt = 0.2

    # Function to compute analytical solution at a specific point (x, t)
    function u_analytic_point(x, t)
        z = -x / 2 + t
        return 11 + 15 * tanh(z) - 15 * tanh(z)^2 - 15 * tanh(z)^3
    end

    # Function to generate the dataset matrix
    function generate_dataset_matrix(domains, dx, dt, xlim, tlim)
        x_values = xlim[1]:dx:xlim[2]
        t_values = tlim[1]:dt:tlim[2]

        dataset = []

        for t in t_values
            for x in x_values
                u_value = u_analytic_point(x, t)
                push!(dataset, [u_value, x, t])
            end
        end

        return vcat([data' for data in dataset]...)
    end

    # considering sparse dataset from half of x's domain
    datasetpde_new = [generate_dataset_matrix(domains, dx, dt, [-10, 0], [0.0, 1.0])]

    # Adding Gaussian noise with a 0.8 std
    noisydataset_new = deepcopy(datasetpde_new)
    noisydataset_new[1][:, 1] = noisydataset_new[1][:, 1] .+
        (randn(size(noisydataset_new[1][:, 1])) .* 0.8)

    # Neural network
    chain = Lux.Chain(
        Lux.Dense(2, 8, Lux.tanh),
        Lux.Dense(8, 8, Lux.tanh),
        Lux.Dense(8, 1)
    )

    # Discretization for old and new models
    discretization = NeuralPDE.BayesianPINN(
        [chain],
        GridTraining([dx, dt]), param_estim = true, dataset = [noisydataset_new, nothing]
    )

    # let α default to 2.0
    @named pde_system = PDESystem(
        eq,
        bcs,
        domains,
        [x, t],
        [u(x, t)],
        [α],
        defaults = Dict([α => 2.0])
    )

    # neccesarry for loss function construction (involves Operator masking)
    eqs = pde_system.eqs
    Dict_differentials = Dict()
    exps = toexpr.(eqs)
    nullobj = [recur_expression(exp, Dict_differentials) for exp in exps]

    # Dict_differentials is now ;
    # Dict{Any, Any} with 5 entries:
    #   Differential(x)(Differential(x)(u(x, t)))            => diff_5
    #   Differential(x)(Differential(x)(Differential(x)(u(x… => diff_1
    #   Differential(x)(Differential(x)(Differential(x)(Dif… => diff_2
    #   Differential(x)(u(x, t))                             => diff_4
    #   Differential(t)(u(x, t))                             => diff_3

    # using HMC algorithm due to convergence, stability, time of training. (refer to mcmc chain plots)
    # choice of std for objectives is very important
    # pass in Dict_differentials, phystdnew arguments when using the new model

    sol_new = ahmc_bayesian_pinn_pde(
        pde_system,
        discretization;
        draw_samples = 150,
        bcstd = [0.1, 0.1, 0.1, 0.1, 0.1], phynewstd = [0.4],
        phystd = [0.2], l2std = [0.8], param = [Distributions.Normal(2.0, 2)],
        priorsNNw = (0.0, 1.0),
        saveats = [1 / 100.0, 1 / 100.0],
        Dict_differentials = Dict_differentials
    )

    sol_old = ahmc_bayesian_pinn_pde(
        pde_system,
        discretization;
        draw_samples = 150,
        bcstd = [0.1, 0.1, 0.1, 0.1, 0.1],
        phystd = [0.2], l2std = [0.8], param = [Distributions.Normal(2.0, 2)],
        priorsNNw = (0.0, 1.0),
        saveats = [1 / 100.0, 1 / 100.0]
    )

    phi = discretization.phi[1]
    xs,
        ts = [
        infimum(d.domain):dx:supremum(d.domain)
            for (d, dx) in zip(domains, [dx / 10, dt])
    ]
    u_real = [[u_analytic(x, t) for x in xs] for t in ts]

    u_predict_new = [
        [first(pmean(phi([x, t], sol_new.estimated_nn_params[1]))) for x in xs]
            for t in ts
    ]

    diff_u_new = [
        [
                abs(
                    u_analytic(x, t) -
                    first(pmean(phi([x, t], sol_new.estimated_nn_params[1])))
                )
                for x in xs
            ]
            for t in ts
    ]

    u_predict_old = [
        [first(pmean(phi([x, t], sol_old.estimated_nn_params[1]))) for x in xs]
            for t in ts
    ]
    diff_u_old = [
        [
                abs(
                    u_analytic(x, t) -
                    first(pmean(phi([x, t], sol_old.estimated_nn_params[1])))
                )
                for x in xs
            ]
            for t in ts
    ]

    unsafe_comparisons(true)
    @test all(all, [((diff_u_new[i]) .^ 2 .< 0.8) for i in 1:6]) == true
    @test all(all, [((diff_u_old[i]) .^ 2 .< 0.8) for i in 1:6]) == false

    MSE_new = [mean(abs2, diff_u_new[i]) for i in 1:6]
    MSE_old = [mean(abs2, diff_u_old[i]) for i in 1:6]
    @test (MSE_new .< MSE_old) == [1, 1, 1, 1, 1, 1]

    param_new = sol_new.estimated_de_params[1]
    param_old = sol_old.estimated_de_params[1]
    α = 1
    @test abs(param_new - α) < 0.2 * α
    @test abs(param_new - α) < abs(param_old - α)
end
