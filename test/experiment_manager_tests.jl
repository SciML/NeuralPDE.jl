begin
using NeuralPDE
using Distributed
end

begin
@show Distributed.nprocs()
Distributed.addprocs(2)
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
        StructGenerator(:BFGSOptimiser, 1000),
        StructGenerator(:ADAMOptimiser, 1000, 1e-3)
    )
)


hyperparametersweep = StructGeneratorHyperParameterSweep(1, 2, sg)
hyperparameters = generate_hyperparameters(hyperparametersweep)

neuralpde_workers = map(NeuralPDE.NeuralPDEWorker, workers())
experiment_manager = NeuralPDE.ExperimentManager(neuralpde_workers, hyperparameters)
end
#NeuralPDE.initialize_envs(experiment_manager) # eh try this again later maybe

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


@everywhere pde_system = get_pde_system()

@everywhere function get_cb()
    cb = function (p,l)
        println("Current loss is: $l")
        return false
    end
    return cb
end

hyperparam = hyperparameters[1]

#res, phi, pdefunc = NeuralPDE.run_neuralpde(pde_system, hyperparam, get_cb())

#Distributed.rmprocs(workers())

worker_hyperparam_pair = zip(workers(), hyperparameters)
@everywhere cb_func = get_cb()

channels = [RemoteChannel(NeuralPDE.remote_run_neuralpde_with_logs(pde_system, hyperparam, cb_func), id) for (id, hyperparam) in worker_hyperparam_pair]
experiment_manager_log_dir = joinpath(pwd(), "logs", "experiment_manager_test_logs")
if isdir(experiment_manager_log_dir)
    rm(experiment_manager_log_dir, recursive=true)
    mkdir(experiment_manager_log_dir)
end
for (id, channel) in zip(workers(), channels)
    while true
        (dir, file, contents) = take!(channel)
        if dir == "nomoredata"  # this could possibly break but they'd have to be taking log data in dir "nomoredata", not "/nomoredata" and I don't even know if that's possible
            break
        else
            @show dir
            @show file
            split_dir = splitpath(dir)
            local_dir = joinpath(vcat(pwd(), split_dir[4:length(split_dir)]))
            @show local_dir
            mkpath(local_dir)
            fileloc = joinpath(local_dir, file)
            write(fileloc, contents)
        end
    end
end
