struct NNPDENS{C1,C2,O,sde_algorithm} <: NeuralNetDiffEqAlgorithm
    u0::C1
    σᵀ∇u::C2
    opt::O
    sde_algorithm::sde_algorithm
end

NNPDENS(u0,σᵀ∇u;opt=Flux.ADAM(0.1),sde_algorithm=EM()) = NNPDENS(u0,σᵀ∇u,opt,sde_algorithm)

function DiffEqBase.solve(
    prob::TerminalPDEProblem,
    alg::NNPDENS;
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    reltol = 1f-5,
    verbose = false,
    maxiters = 300,
    dt,
    trajectories)

    X0 = prob.X0
    tspan = prob.tspan
    ts = prob.tspan[1]:dt:prob.tspan[2]
    d  = length(X0)
    g,f,μ,σ,p = prob.g,prob.f,prob.μ,prob.σ,prob.p

    data = Iterators.repeated((), maxiters)


    #hidden layer
    opt = alg.opt
    u0 = alg.u0
    σᵀ∇u = alg.σᵀ∇u
    sde_algorithm = alg.sde_algorithm
    ps = Flux.params(u0, σᵀ∇u...)

    function F(h, p, t)
        u =  h[end]
        X =  h[1:end-1].data
        _σᵀ∇u = σᵀ∇u([X;t])
        _f = -f(X, u, _σᵀ∇u, p, t)
        Flux.Tracker.collect([μ(X,p,t);[_f]])
    end

    function G(h, p, t)
        X = h[1:end-1].data
        _σᵀ∇u = σᵀ∇u([X;t])'
        Flux.Tracker.collect([σ(X,p,t);_σᵀ∇u])
    end

    function neural_sde(init_cond, F, G, tspan, args...; kwargs...)
        noise = Flux.Tracker.collect(zeros(Float32,d+1,d))
        prob = SDEProblem(F, G, init_cond, tspan, noise_rate_prototype=noise)
        map(1:trajectories) do j #TODO add Ensemble Simulation
            predict_ans = solve(prob,  args...; kwargs...)[end]
            (X,u) = (predict_ans[1:end-1], predict_ans[end])
        end
    end

    n_sde = init_cond->neural_sde(init_cond,F,G,tspan,sde_algorithm, dt=dt,
                                    saveat=ts,abstol=abstol,reltol=reltol)

    function predict_n_sde()
        _u0 = u0(X0)
        init_cond = Flux.Tracker.collect([X0;_u0])
        n_sde(init_cond)
    end

    function loss_n_sde()
        mean(sum(abs2, g(X.data) - u) for (X,u) in predict_n_sde())
    end

    cb = function ()
        l = loss_n_sde()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    Flux.train!(loss_n_sde, ps, data, opt; cb = cb)

    u0(X0)[1].data
end #pde_solve_ns
