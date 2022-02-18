
function run_neuralpde(pde_system::PDESystem, hyperparam::AbstractHyperParameter, cb_func; logger=nothing)
    #@show hyperparam

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

function remote_run_neuralpde_with_logs(pde_system::PDESystem, hyperparam::AbstractHyperParameter, cb_func, experiment_index)
    loggerloc = joinpath(homedir(), "logs", "experiment_manager_test_logs", "$experiment_index")

    # TODO: this is mostly for testing iteration, remove this (in actual use you'd want to make more than one run w/ same hyperparams visible under the same namespace)
    if isdir(loggerloc)
        rm(loggerloc, recursive=true)
    end
    logger = TBLogger(loggerloc, tb_append) #create tensorboard logger

    res, phis, pdefunc = run_neuralpde(pde_system, hyperparam, cb_func; logger=logger)

    # transfer the log files at the end of the run.  this could probably be done more elegantly, with scp or rsync or something,
    # but this only requires a julia worker connection, and it uses the tunneled ssh connection, so it will work behind firewalls
    # and whatnot if you're able to connect past it
    log_vector = Vector{Tuple{String, String, String}}()
    for (root, _, files) in walkdir(loggerloc)
        for file in files
            fileloc = joinpath(root, file)
            filecontents = read(fileloc, String)
            push!(log_vector, (root, file, filecontents))
        end
    end
    log_vector
end

struct NeuralPDEWorker
    pid::Int64
    has_gpu::Bool # currently unused but I've defined it so that it's easy to use it later
    SciMLBase.@add_kwonly function NeuralPDEWorker(pid::Integer; has_gpu=false)  # default to no gpu for workers
        new(convert(Int64, pid), convert(Bool, has_gpu))
    end
end


struct ExperimentInProgress{H <: AbstractHyperParameter}
    hyperparam::H
    future::Future
end

struct ExperimentManager{H <: AbstractHyperParameter, C}
    pde_system::PDESystem
    hyperparameter_queue::Queue{H} 
    cb_func::C
    workers::Vector{NeuralPDEWorker}
    experiments_in_progress::Vector{Union{ExperimentInProgress{H}, Nothing}}
    SciMLBase.@add_kwonly function ExperimentManager(pde_system::PDESystem, hyperparameters::Vector{H}, 
        cb_func::C, workers::Vector{NeuralPDEWorker}) where {H <: AbstractHyperParameter, C}
        hyperparameter_queue = Queue{H}()
        for hyperparameter in hyperparameters
            enqueue!(hyperparameter_queue, hyperparameter)
        end
        num_workers = length(workers)
        nothing_in_progress = Vector{Union{ExperimentInProgress{H}, Nothing}}(nothing, num_workers)
        new{H, C}(pde_system, hyperparameter_queue, cb_func, workers, nothing_in_progress)
    end
end


function run_experiment_queue(experiment_manager::ExperimentManager{H}) where {H <: AbstractHyperParameter}
    pde_system = experiment_manager.pde_system
    cb_func = experiment_manager.cb_func
    experiments_in_progress = experiment_manager.experiments_in_progress
    hyperparameter_queue = experiment_manager.hyperparameter_queue
    workers = experiment_manager.workers
    num_workers = length(experiment_manager.workers)
    println("executing $(length(hyperparameter_queue)) experiments")
    experiment_index = 1

    while true
        # first check for break condition
        no_experiments_running = all(map(isnothing, experiments_in_progress))
        #@show no_experiments_running
        num_hyperparameters_in_queue = length(hyperparameter_queue)
        #@show num_hyperparameters_in_queue
        if no_experiments_running && num_hyperparameters_in_queue == 0 # all done!
            break
        end

        # then iterate through the free workers and assign them an experiment and remote call it
        #println("assigning experiments")
        for worker_index in 1:num_workers
            # free worker and an unassigned hyperparameter, give it a job
            if experiments_in_progress[worker_index] isa Nothing && num_hyperparameters_in_queue > 0
                pid = workers[worker_index].pid
                # grab next hyperparameter and maintain the count 
                hyperparam = dequeue!(hyperparameter_queue)
                num_hyperparameters_in_queue -= 1

                # remote call and store Future for the logs
                future = Future()
                errormonitor(@async put!(future, remotecall_fetch(NeuralPDE.remote_run_neuralpde_with_logs, pid, pde_system, hyperparam, cb_func, experiment_index)))
                experiment_index += 1
                println("assigned experiment $experiment_index to worker $(workers[worker_index].pid)")
                println("$num_hyperparameters_in_queue hyperparameters left in queue")

                # change experiment_in_progress data structure
                experiments_in_progress[worker_index] = ExperimentInProgress{H}(hyperparam, future)
            end
        end


        #println("checking for completed experiments")
        # then gather results of finished experiments and clean up experiment_in_progress data structure
        for worker_index in 1:num_workers
            if !(experiments_in_progress[worker_index] isa Nothing)
                future = experiments_in_progress[worker_index].future
                if isready(future) # the experiment is done, get all the data out
                    worker_pid = workers[worker_index].pid
                    println("experiment done! reading future from worker $worker_pid")
                    log_vector = fetch(future)
                    for (dir, file, contents) in log_vector
                        @show dir
                        @show file
                        split_dir = splitpath(dir)
                        local_dir = joinpath(vcat(pwd(), split_dir[4:length(split_dir)]))
                        @show local_dir
                        mkpath(local_dir)
                        fileloc = joinpath(local_dir, file)
                        write(fileloc, contents)
                    end
                    # clean up data structures 
                    experiments_in_progress[worker_index] = nothing
                end
            end
        end
    end
    println("experiments all done!")
end



