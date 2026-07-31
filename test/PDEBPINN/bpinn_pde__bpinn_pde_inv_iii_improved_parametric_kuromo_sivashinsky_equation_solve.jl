using NeuralPDE
using Test

@testset "BPINN PDE Inv III: Improved Parametric Kuromo-Sivashinsky Equation solve" begin
    using MCMCChains, Lux, ModelingToolkit, Distributions, OrdinaryDiffEq,
        AdvancedHMC, LogDensityProblems, Statistics, Random, Functors, NeuralPDE, MonteCarloMeasurements,
        ComponentArrays, SymbolicUtils
    import DomainSets: Interval, infimum, supremum

    Random.seed!(100)

    function recur_expression(term, Dict_differentials)
        term = SymbolicUtils.unwrap(term)
        SymbolicUtils.iscall(term) || return nothing

        op = SymbolicUtils.operation(term)
        if op isa ModelingToolkit.Differential
            Dict_differentials[term] = Symbolics.variable("diff_$(length(Dict_differentials) + 1)")
            return nothing
        end

        for arg in SymbolicUtils.arguments(term)
            recur_expression(arg, Dict_differentials)
        end
        return nothing
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
        initial_conditions = Dict([α => 2.0])
    )

    # neccesarry for loss function construction (involves Operator masking)
    eqs = pde_system.eqs
    Dict_differentials = Dict()
    for eq in eqs
        recur_expression(eq.lhs, Dict_differentials)
        recur_expression(eq.rhs, Dict_differentials)
    end

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

    MSE_new = [mean(abs2, diff_u_new[i]) for i in 1:6]
    MSE_old = [mean(abs2, diff_u_old[i]) for i in 1:6]
    @test mean(MSE_new) < mean(MSE_old) + 0.5

    param_new = sol_new.estimated_de_params[1]
    param_old = sol_old.estimated_de_params[1]
    α = 1
    @test abs(param_new - α) < 0.8 * α
end
