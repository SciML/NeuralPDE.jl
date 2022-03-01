using Pkg
Pkg.activate(ENV["NEURALPDE_DIR"]) 
begin
using DiffEqBase
using DelimitedFiles
using CSV
using Plots
using DataFrames

using Random
import ModelingToolkit: Interval, infimum, supremum
using Logging, TensorBoardLogger
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
println("loaded packages!")
end
function rm_subdirs(dir)
  map(readdir(dir; join=true)) do subdir
    println("rming subdir $subdir")
    rm(subdir;recursive=true) 
  end
end

receive_logdir = joinpath(["logs", "spm_hyperparam_tests"])
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

testdir = joinpath([homedir(), ".julia", "dev", "NeuralPDE.jl", "test"])
sols_t = Array(CSV.read(joinpath([testdir, "sols_t.csv"]), DataFrame; header=false))[:,1]
sols_r_n = Array(CSV.read(joinpath([testdir, "sols_r_n.csv"]), DataFrame; header=false))[:,1]
sols_r_p = Array(CSV.read(joinpath([testdir, "sols_r_p.csv"]), DataFrame; header=false))[:,1]
sols_c_s_n = Array(CSV.read(joinpath([testdir, "sols_c_s_n.csv"]), DataFrame; header=false))
sols_c_s_p = Array(CSV.read(joinpath([testdir, "sols_c_s_p.csv"]), DataFrame; header=false))



sg = StructGenerator(
    :CompositeHyperParameter,
    RandomChoice(1:2^10), # seed
    StructGenerator( # nn
        :SimpleFeedForwardNetwork, # type/constructor name
        RandomChoice(2:4),
        RandomChoice(30, 40, 50),
        RandomChoice(:GELUNonLin, :SigmoidNonLin),
        :GlorotUniformParams
    ),
    StructGenerator( # training
        :StochasticTraining,
        RandomChoice(32, 64, 128)
    ),
    RandomChoice( # adaptive loss
        StructGenerator(
            :GradientScaleAdaptiveLoss,
            50
        ),
        StructGenerator(
            :MiniMaxAdaptiveLoss,
            50
        )
    ),
    RandomChoice( # optimizer
        StructGenerator(:ADAMOptimiser, 10000, 1e-2),
        StructGenerator(:ADAMOptimiser, 10000, 3e-3),
        StructGenerator(:ADAMOptimiser, 10000, 1e-3),
        StructGenerator(:ADAMOptimiser, 10000, 3e-4),
    )
)


hyperparametersweep = StructGeneratorHyperParameterSweep(1, num_hyperparameters, sg)
hyperparameters = generate_hyperparameters(hyperparametersweep)

println("made hyperparams")

function get_pde_system()

    # ('negative particle',) -> rn
    # ('positive particle',) -> rp
    @parameters t rn rp
    # 'Discharge capacity [A.h]' -> Q
    # 'X-averaged negative particle concentration' -> c_s_n_xav
    # 'X-averaged positive particle concentration' -> c_s_p_xav
    @variables Q(..) c_s_n_xav(..) c_s_p_xav(..)
    #@variables Q(t) c_s_n_xav(t, rn) c_s_p_xav(t, rp)
    Dt = Differential(t)
    Drn = Differential(rn)
    Drp = Differential(rp)
    # 'X-averaged negative particle concentration' equation
    #cache_4647021298618652029 = 8.813457647415216 * (1 / rn^2 * Drn(rn^2 * Drn(c_s_n_xav(t, rn))))
    cache_4647021298618652029 = 8.813457647415216 * (Drn(Drn(c_s_n_xav(t, rn))) + 2 / rn * Drn(c_s_n_xav(t, rn)))

    # 'X-averaged positive particle concentration' equation
    #cache_m620026786820028969 = 22.598609352346717 * (1 / rp^2 * Drp(rp^2 * Drp(c_s_p_xav(t, rp))))
    cache_m620026786820028969 = 22.598609352346717 * (Drp(Drp(c_s_p_xav(t, rp))) + 2 / rp * Drp(c_s_p_xav(t, rp)))


    eqs = [
    Dt(Q(t)) ~ 4.27249308415467,
    Dt(c_s_n_xav(t, rn)) ~ cache_4647021298618652029,
    Dt(c_s_p_xav(t, rp)) ~ cache_m620026786820028969,
    ]

    ics_bcs = [
    Q(0) ~ 0.0,
    c_s_n_xav(0, rn) ~ 0.8000000000000016,
    c_s_p_xav(0, rp) ~ 0.6000000000000001,
    Drn(c_s_n_xav(t, 0.01)) ~ 0.0,
    Drn(c_s_n_xav(t, 1.0)) ~ -0.14182855923368468,
    Drp(c_s_p_xav(t, 0.01)) ~ 0.0,
    Drp(c_s_p_xav(t, 1.0)) ~ 0.03237700710041634,
    ]

    t_domain = IntervalDomain(0.0, 0.15) # 0.15
    rn_domain = IntervalDomain(0.01, 1.0)
    rp_domain = IntervalDomain(0.01, 1.0)

    domains = [
    t in t_domain,
    rn in rn_domain,
    rp in rp_domain,
    ]
    ind_vars = [t, rn, rp]
    dep_vars = [Q(t), c_s_n_xav(t, rn), c_s_p_xav(t, rp)]

    @named SPM_pde_system = PDESystem(eqs, ics_bcs, domains, ind_vars, dep_vars)
end


pde_system = get_pde_system()

function get_cb()
    cb = function (p,l)
        return false
    end
    return cb
end

cb_func = get_cb()
function get_plot_function()

    p1 = plot(sols_t, sols_r_n, sols_c_s_n, linetype=:contourf,xlabel="t", ylabel="r_n", title = "pybamm c_s_n")
    p2 = plot(sols_t, sols_r_p, sols_c_s_p, linetype=:contourf,xlabel="t", ylabel="r_p", title = "pybamm c_s_p")
    function plot_function(logger, step, phi, θ, indices_in_params, adaloss)
        phi_i_params = [θ[indices_in_params_i] for indices_in_params_i in indices_in_params]

        #Q_evals = [chains_[1]([dt], phi_i_params[1])[1] for dt in dts]

        c_s_n_evals = [phi[2]([dt, drn], phi_i_params[2])[1] for drn in sols_r_n, dt in sols_t]
        c_s_p_evals = [phi[3]([dt, drp], phi_i_params[3])[1] for drp in sols_r_p, dt in sols_t]
        c_s_n_error = abs.(c_s_n_evals .- sols_c_s_n)
        c_s_p_error = abs.(c_s_p_evals .- sols_c_s_p)

        c_s_n_error_sum = sum(c_s_n_error)
        c_s_p_error_sum = sum(c_s_p_error)
        c_s_n_error_relative = c_s_n_error_sum / sum(abs.(sols_c_s_n))
        c_s_p_error_relative = c_s_p_error_sum / sum(abs.(sols_c_s_p))
        log_value(logger, "full_c_s_n_error", c_s_n_error_sum; step=step)
        log_value(logger, "full_c_s_p_error", c_s_p_error_sum; step=step)
        log_value(logger, "full_c_s_n_error_relative", c_s_n_error_relative; step=step)
        log_value(logger, "full_c_s_p_error_relative", c_s_p_error_relative; step=step)


        p3 = plot(sols_t, sols_r_n, c_s_n_evals, linetype=:contourf,xlabel="t", ylabel="r_n", title = "learned c_s_n")
        p4 = plot(sols_t, sols_r_p, c_s_p_evals, linetype=:contourf,xlabel="t", ylabel="r_p", title = "learned c_s_p")
        p5 = plot(sols_t, sols_r_n, c_s_n_error, linetype=:contourf,xlabel="t", ylabel="r_n", title = "c_s_n error")
        p6 = plot(sols_t, sols_r_p, c_s_p_error, linetype=:contourf,xlabel="t", ylabel="r_p", title = "c_s_p error")
        [(name="pybamm c_s_n", image=p1), (name="pybamm c_s_p", image=p2),
         (name="learned c_s_n", image=p3), (name="learned c_s_p", image=p4),
         (name="c_s_n error", image=p5), (name="c_s_p error", image=p6)]
    end
    return plot_function
end
plot_function = get_plot_function()
log_options = LogOptions(;plot_function=plot_function, plot_frequency=500)

println("structures complete")


for i in hyperparameter_indices_to_compute
	this_logdir = joinpath([receive_logdir, string(i)])
	logger = TBLogger(this_logdir, tb_append) #create tensorboard logger
	println("made logger at $(this_logdir), starting exp $(i)")

	@show hyperparameters[i]

	NeuralPDE.run_neuralpde(pde_system, hyperparameters[i], cb_func; logger=logger, log_options=log_options)
end

println("all done with experiments")

