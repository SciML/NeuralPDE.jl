"""
@everywhere begin
    using Pkg
    Pkg.activate(".")
    println(Pkg.installed())
end
"""

using Distributed
#@everywhere using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
#@everywhere import ModelingToolkit: Interval, infimum, supremum

#@show Distributed.nprocs()
#@show Threads.nthreads()

i = t = id = 0



function run_neuralpde(pde_system::PDESystem, hyperparam::AbstractHyperParameter, cb_func; logger=nothing)
    @show hyperparam

    iteration = [0]
    function wrapped_iteration_cb_func(p, l) # useful for making sure the user doesn't need to iterate
        iteration[1] += 1
        cb_func(p, l)
    end

    seed = NeuralPDE.getseed(hyperparam)
    Random.seed!(seed)

    # Neural network
    num_ivs_for_dvs = map(pde_system.dvs) do dv
        # assumes dv is in the form u(t,x) etc 
        num_iv_for_dv = length(dv.val.arguments)
    end
    chains, init_params = NeuralPDE.getfunction(hyperparam, num_ivs_for_dvs)

    training = NeuralPDE.gettraining(hyperparam)

    discretization = PhysicsInformedNN(chains, training; init_params=init_params, logger=logger, iteration=iteration)
    prob = discretize(pde_system,discretization)

    # Optimizer
    opt, maxiters = NeuralPDE.getopt(hyperparam)

    res = GalacticOptim.solve(prob,opt; cb = wrapped_iteration_cb_func, maxiters=maxiters)
    phis = discretization.phi
    return (res=res, phis=phis, pdefunc=tx->map(phi->phi(tx, res)[1], phis) )
end

"""

begin
end
begin
    res, phi, pdefunc = run_neuralpde(get_pde_system(), hyperparam, get_cb())
    @show pdefunc([0.0, 0.5])
end
"""

# distributed experiment queue

"""
@everywhere workers() begin; using Pkg; Pkg.activate("."); Pkg.update(); Pkg.add(["Distributed", "JSON", "DiffEqBase", "TensorBoardLogger", "Logging", "NeuralPDE", "ModelingToolkit", "Symbolics", "DiffEqFlux", "Flux", "Parameters", "ImageCore"]); Pkg.instantiate(); end
@everywhere workers() begin; using Pkg; Pkg.activate("."); Pkg.instantiate(); using Logging, TensorBoardLogger, NeuralPDE, ModelingToolkit, Symbolics, DiffEqFlux, Flux, Parameters; end

@everywhere workers() @show Pkg.project()

#@everywhere workers() begin; mkdir("$(myid())"); end
"""

function remote_run_neuralpde_with_logs(pde_system::PDESystem, hyperparam::AbstractHyperParameter, cb_func)
    function inner_run_neuralpde_with_logs()
        id = myid()
        loggerloc = joinpath(homedir(), "logs", "experiment_manager_test_logs", "$id")
        if isdir(loggerloc)
            rm(loggerloc, recursive=true)
        end
        logger = TBLogger(loggerloc, tb_append) #create tensorboard logger

        res, phis, pdefunc = run_neuralpde(pde_system, hyperparam, cb_func; logger=logger)

        """
        ################log scalars example: y = x²################
        #using logger interface
        #with_logger(logger) do
        for x in 1:20
            #@info "scalar/loggerinterface" y = x*x *(-id)
            log_value(logger, "scalar/loggerinterface", x*x *(id) * params; step=x)
        end
        #end
        #using explicit function interface
        for x in 1:20
            log_value(logger, "scalar/explicitinterface", x*x*id; step = x)
        end


        ################log scalar example: y = x-xi################
        #with_logger(logger) do
        for x in 1:20
            z = x-x*im + id - im*id
            #@info "scalar/complex" y = z
            log_value(logger, "scalar/complex", z; step=x)
        end
        #end
        """

        function producer(c::Channel)
            for (root, _, files) in walkdir(loggerloc)
                for file in files
                    fileloc = joinpath(root, file)
                    filecontents = read(fileloc, String)
                    put!(c, (root, file, filecontents))
                end
            end
            put!(c, ("nomoredata", "", ""))
        end

        Channel{Tuple{String, String, String}}(producer)
    end
    inner_run_neuralpde_with_logs
end

struct NeuralPDEWorker
    pid::Int64
    has_gpu::Bool
    SciMLBase.@add_kwonly function NeuralPDEWorker(pid::Integer; has_gpu=false)  # default to no gpu for workers
        new(convert(Int64, pid), convert(Bool, has_gpu))
    end
end


struct ExperimentInProgress{H <: AbstractHyperParameter}
    hyperparam::H
    remote_channel::RemoteChannel{Channel{Tuple{String, String, String}}}
end

struct ExperimentManager{H <: AbstractHyperParameter}
    workers::Vector{NeuralPDEWorker}
    experiment_queue::Queue{H} 
    experiments_in_progress::Vector{Union{ExperimentInProgress{H}, Nothing}}
    SciMLBase.@add_kwonly function ExperimentManager(workers::Vector{NeuralPDEWorker}, experiment_vector::Vector{H}) where H <: AbstractHyperParameter
        experiment_queue = Queue{H}()
        for experiment in experiment_vector
            enqueue!(experiment_queue, experiment)
        end
        num_workers = length(workers)
        nothing_in_progress = Vector{Union{ExperimentInProgress{H}, Nothing}}(nothing, num_workers)
        new{H}(workers, experiment_queue, nothing_in_progress)
    end
end



"""
# eh, just assume that the top level script activates and imports the right packages for now. this is hard
function initialize_envs(experiment_manager::ExperimentManager, env_string::AbstractString)
    activated_results_futures = map(experiment_manager.pids) do pid
        remotecall(worker_initialize_env, pid, env_string)
    end
    activated_results = map(activated_results_futures) do result_future
        fetch(result_future)
    end
    activated_results
end


function worker_initialize_env(env_string::AbstractString)
    try 
        Pkg.activate(env_string)
    catch e
        return e
    end
    true
end
"""

