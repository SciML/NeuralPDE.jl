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


# run a single experiment


function get_discretization_opt_maxiters(pde_system::PDESystem, hyperparam::AbstractHyperParameter)
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


function run_neuralpde(pde_system::PDESystem, hyperparam::AbstractHyperParameter, cb_func)
    @show hyperparam
    seed = NeuralPDE.getseed(hyperparam)
    Random.seed!(seed)
    discretization, opt, maxiters = get_discretization_opt_maxiters(pde_system, hyperparam)

    prob = discretize(pde_system,discretization)


    res = GalacticOptim.solve(prob,opt; cb = cb_func, maxiters=maxiters)
    phis = discretization.phi
    return (res=res, phis=phis, pdefunc=tx->map(phi->phi(tx, res)[1], phis)   )
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

@everywhere function loggerdata(params)
    function loggerdataparams()
        loggerloc = joinpath("scalarlogs", "$(myid())")
        if isdir(loggerloc)
            rm(loggerloc, recursive=true)
        end
        id = myid()
        logger = TBLogger(loggerloc, tb_append) #create tensorboard logger

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
    loggerdataparams
end


channels = [RemoteChannel(loggerdata(id - 1), id) for id in workers()]
if isdir("scalarlogs")
    rm("scalarlogs", recursive=true)
end
for (id, channel) in zip(workers(), channels)
    while true
        (dir, file, contents) = take!(channel)
        if dir == "nomoredata"
            break
        else
            mkpath(dir)
            fileloc = joinpath(dir, file)
            write(fileloc, contents)
        end
    end
end


log1 = "1/scalarlogs/events.out.tfevents.1.639904028536489e9.hecate"
read(log1, String)



@everywhere function sleeprandom()
    t = rand()
    sleep(t)
    t
end

begin
    c = counter(Int)

    q = Queue{Int}()
    for i in 1:30
        enqueue!(q, i)
    end

    futures = Dict{Int, Future}()

    for id in workers()
        if length(q) > 0
            i = dequeue!(q)
            futures[id] = remotecall(sleeprandom, id)
            inc!(c, id)
            println("worker $id assigned parcel $i")
        end
    end

    while length(q) > 0
        for id in workers()
            if isready(futures[id])
                t = fetch(futures[id])
                println("worker $id finished a parcel in $t seconds")
                if length(q) > 0
                    i = dequeue!(q)
                    futures[id] = remotecall(sleeprandom, id)
                    inc!(c, id)
                    println("worker $id assigned parcel $i")
                end
            end
        end
    end

    @show c
end
"""


struct ExperimentManager
    workers::Vector{NeuralPDEWorker}
    experiment_queue::Queue{AbstractHyperParameter}
    experiments_in_progress::Vector{Tuple{NeuralPDEWorker, AbstractHyperParameter, RemoteChannel{Tuple{String, String, String}}}}
    SciMLBase.@add_kwonly function ExperimentManager(workers::Vector{NeuralPDEWorker}, experiment_vector::Vector{AbstractHyperParameter})
        experiment_queue = Queue{AbstractHyperParameter}()
        empty_in_progress = Vector{Tuple{NeuralPDEWorker, AbstractHyperParameter, RemoteChannel{Tuple{String, String, String}}}}()
        new(workers, experiment_queue, empty_in_progress)
    end
end

struct NeuralPDEWorker
    pid::Int64
    has_gpu::Bool
    SciMLBase.@add_kwonly function NeuralPDEWorker(pid::Integer; has_gpu=false)
        new(convert(Int64, log_frequency), convert(Bool, has_gpu))
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

