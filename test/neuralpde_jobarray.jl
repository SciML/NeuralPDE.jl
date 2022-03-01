# Activate environment
using Pkg
Pkg.activate(ENV["NEURALPDE_DIR"]) 

function rm_subdirs(dir)
  map(readdir(dir; join=true)) do subdir
    println("rming subdir $subdir")
    rm(subdir;recursive=true) 
  end
end

receive_logdir = joinpath(["logs", "experiment_manager_test_logs"])
mkpath(receive_logdir)
rm_subdirs(receive_logdir)

@show ARGS
task_id = parse(Int,ARGS[1])
num_tasks = parse(Int,ARGS[2])
num_hyperparameters = 64
num_hyperparameters_per_task = Int(ceil(num_hyperparameters/num_tasks))
@show num_hyperparameters_per_task
hyperparameter_indices_to_compute = range((task_id - 1) * num_hyperparameters_per_task + 1, min(task_id * num_hyperparameters_per_task, num_hyperparameters))
@show hyperparameter_indices_to_compute


println("hi from job $(task_id) of $(num_tasks), pwd: $(pwd()), homedir: $(homedir())")

# Load required packages and helpers on all processes
begin
  import ModelingToolkit: Interval, infimum, supremum
  using Logging, TensorBoardLogger
  using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
  println("loaded packages!")
end

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
        StructGenerator(:ADAMOptimiser, 10000, 1e-2),
        StructGenerator(:ADAMOptimiser, 10000, 1e-3)
    )
)


hyperparametersweep = StructGeneratorHyperParameterSweep(1, num_hyperparameters, sg)
hyperparameters = generate_hyperparameters(hyperparametersweep)

println("made hyperparams")

function get_pde_system()

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

function get_cb()
    cb = function (p,l)
        return false
    end
    return cb
end

cb_func = get_cb()

println("structures complete")

# TODO: this is mostly for testing iteration, remove this (in actual use you'd want to make more than one run w/ same hyperparams visible under the same namespace)


for i in hyperparameter_indices_to_compute
	this_logdir = joinpath([receive_logdir, string(i)])
	logger = TBLogger(this_logdir, tb_append) #create tensorboard logger
	println("made logger at $(this_logdir), starting exp $(i)")

	@show hyperparameters[i]

	NeuralPDE.run_neuralpde(pde_system, hyperparameters[i], cb_func; logger=logger)
end

println("all done with experiments")

