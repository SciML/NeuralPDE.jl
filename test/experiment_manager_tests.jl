begin
using NeuralPDE
using Distributed
using Plots
function rm_subdirs(dir)
    map(readdir(dir; join=true)) do subdir
        println("rming subdir $subdir")
        rm(subdir;recursive=true) 
    end
end
receive_logdir = joinpath(["logs", "experiment_manager_test_logs"])
remote_logdir = joinpath([homedir(), "logs", "experiment_manager_test_logs"])
mkpath(receive_logdir)
rm_subdirs(receive_logdir)
mkpath(remote_logdir)
rm_subdirs(remote_logdir)
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
@everywhere using Logging, TensorBoardLogger, ImageIO, ImageMagick
@everywhere using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
@everywhere using Plots
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
        StructGenerator(:ADAMOptimiser, 10000, 1e-2),
        StructGenerator(:ADAMOptimiser, 10000, 1e-3)
    )
)


hyperparametersweep = StructGeneratorHyperParameterSweep(1, 2, sg)
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

    @named pde_system = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])

    return (pde_system=pde_system, domains=domains)
end


pde_system, domains = get_pde_system()

@everywhere function get_cb()
    cb = function (p,l)
        return false
    end
    return cb
end

@everywhere function get_plot_function()
    xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
    analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)
    u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
    p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
    function plot_function(phi, θ, adaloss)
        u_predict = reshape([first(phi[1]([x,y],θ)) for x in xs for y in ys],(length(xs),length(ys)))
        diff_u = abs.(u_predict .- u_real)


        p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
        p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
        [(name="analytic", image=p1), (name="predict", image=p2), (name="error", image=p3)]
    end
    return plot_function
end

log_options = NeuralPDE.LogOptions(;plot_function=get_plot_function())

neuralpde_workers = map(NeuralPDE.NeuralPDEWorker, workers())
cb_func = get_cb()
end
experiment_manager = NeuralPDE.ExperimentManager(pde_system, hyperparameters, cb_func, log_options, neuralpde_workers)


NeuralPDE.run_experiment_queue(experiment_manager)

#res, phi, pdefunc = NeuralPDE.run_neuralpde(pde_system, hyperparam, get_cb())

#Distributed.rmprocs(workers())

