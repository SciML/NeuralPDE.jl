begin
using NeuralPDE
using Distributed
end

begin
@show Distributed.nprocs()
Distributed.addprocs(4)
@show Distributed.nprocs()
test_env = pwd()
end

begin
@everywhere workers() begin; using Pkg; Pkg.activate($test_env); end
@everywhere import ModelingToolkit: Interval, infimum, supremum
@everywhere using Logging, TensorBoardLogger
@everywhere using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
@everywhere workers() @show Pkg.project()
@everywhere workers() @show pwd()
@everywhere workers() @show homedir()
end
begin


sg = StructGenerator(
    :CompositeHyperParameter,
    RandomChoice(1:2^10), # seed
    StructGenerator( # nn
        :SimpleFeedForwardNetwork, # type/constructor name
        RandomChoice(1:2),
        RandomChoice(10, 20, 30),
        RandomChoice(:GELUNonLin, :SigmoidNonLin),
        :GlorotUniformParams
    ),
    StructGenerator( # training
        :GridTraining,
        RandomChoice(0.1, 0.2, 0.06)
    ),
    RandomChoice( # optimizer
        StructGenerator(:BFGSOptimiser, 10000),
        StructGenerator(:ADAMOptimiser, 10000, 1e-3)
    )
)


hyperparametersweep = StructGeneratorHyperParameterSweep(1, 16, sg)
hyperparameters = generate_hyperparameters(hyperparametersweep)


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


pde_system = get_pde_system()

@everywhere function get_cb()
    cb = function (p,l)
        return false
    end
    return cb
end

neuralpde_workers = map(NeuralPDE.NeuralPDEWorker, workers())
cb_func = get_cb()
end
experiment_manager = NeuralPDE.ExperimentManager(pde_system, hyperparameters, cb_func, neuralpde_workers)


NeuralPDE.run_experiment_queue(experiment_manager)

#res, phi, pdefunc = NeuralPDE.run_neuralpde(pde_system, hyperparam, get_cb())

#Distributed.rmprocs(workers())

