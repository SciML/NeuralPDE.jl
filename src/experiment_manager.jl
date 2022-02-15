"""
@everywhere begin
    using Pkg
    Pkg.activate(".")
    println(Pkg.installed())
end
"""

using Distributed
@everywhere using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
@everywhere import ModelingToolkit: Interval, infimum, supremum

@show Distributed.nprocs()
@show Threads.nthreads()

@everywhere function get_pde_system()

    @parameters t, x
    @variables u(..)
    Dxx = Differential(x)^2
    Dtt = Differential(t)^2
    Dt = Differential(t)

    #2D PDE
    C=1
    eq  = Dtt(u(t,x)) ~ C^2*Dxx(u(t,x))

    # Initial and boundary conditions
    bcs = [u(t,0) ~ 0.,# for all t > 0
        u(t,1) ~ 0.,# for all t > 0
        u(0,x) ~ x*(1. - x), #for all 0 < x < 1
        Dt(u(0,x)) ~ 0. ] #for all  0 < x < 1]

    # Space and time domains
    domains = [t ∈ Interval(0.0,1.0),
            x ∈ Interval(0.0,1.0)]

    return @named pde_system = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])
end

@everywhere function get_discretization_opt_maxiters(pde_system::PDESystem, hyperparam::AbstractHyperParameter)
    # Neural network
    num_ivs = length(pde_system.ivs)
    num_ivs_for_dvs = map(pde_system.dvs) do dv
        # assumes dv is in the form u(t,x) etc 
        num_iv_for_dv = length(dv.val.arguments)
    end
    chains, init_params = NeuralPDE.getfunction(hyperparam, num_ivs_for_dvs)
    #if length(chains) == 1
        #chains = chains[1]
    #end

    training = NeuralPDE.gettraining(hyperparam)
    

    discretization = PhysicsInformedNN(chains, training; init_params=init_params)

    # Optimiser
    opt, maxiters = NeuralPDE.getopt(hyperparam)

    return (discretization=discretization, opt=opt, maxiters=maxiters)
end

@everywhere function get_cb()
    cb = function (p,l)
        println("Current loss is: $l")
        return false
    end
    return cb
end

@everywhere function run_neuralpde(pde_system::PDESystem, hyperparam::AbstractHyperParameter, cb_func)
    @show hyperparam
    seed = NeuralPDE.getseed(hyperparam)
    Random.seed!(seed)
    discretization, opt, maxiters = get_discretization_opt_maxiters(pde_system, hyperparam)

    prob = discretize(pde_system,discretization)


    res = GalacticOptim.solve(prob,opt; cb = cb_func, maxiters=maxiters)
    phis = discretization.phi
    return (res=res, phis=phis, pdefunc=tx->map(phi->phi(tx, res)[1], phis)   )
end


begin
    sg = StructGenerator(
        :CompositeHyperParameter,
        RandomChoice(1:2^10), # seed
        StructGenerator( # nn
            :SimpleFeedForwardNetwork, # type/constructor name
            RandomChoice(3:6),
            RandomChoice(32, 64, 128),
            RandomChoice(:GELUNonLin, :SigmoidNonLin),
            :GlorotUniformParams
        ),
        StructGenerator( # training
            :GridTraining,
            RandomChoice(0.1, 0.04, 0.02)
        ),
        RandomChoice( # optimizer
            StructGenerator(:BFGSOptimiser, 1000),
            StructGenerator(:ADAMOptimiser, 1000, 1e-3)
        )
    )


    hyperparametersweep = StructGeneratorHyperParameterSweep(1, 16, sg)
    hyperparameters = generate_hyperparameters(hyperparametersweep)
    hyperparam = hyperparameters[1]
end
begin
    res, phi, pdefunc = run_neuralpde(get_pde_system(), hyperparam, get_cb())
    @show pdefunc([0.0, 0.5])
end