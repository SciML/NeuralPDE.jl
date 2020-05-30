struct NNPDENS{C1,C2,O} <: NeuralNetDiffEqAlgorithm
    u0::C1
    σᵀ∇u::C2
    opt::O
end

NNPDENS(u0,σᵀ∇u;opt=Flux.ADAM(0.1)) = NNPDENS(u0,σᵀ∇u,opt)

function DiffEqBase.solve(
    prob::TerminalPDEProblem,
    pdealg::NNPDENS;
    verbose = false,
    maxiters = 300,
    trajectories = 100,
    alg,
    pabstol = 1f-6,
    save_everystep = false,
    kwargs...)

    X0 = prob.X0
    tspan = prob.tspan
    d  = length(X0)
    g,f,μ,σ = prob.g,prob.f,prob.μ,prob.σ
    p = prob.p isa AbstractArray ? prob.p : Float32[]

    data = Iterators.repeated((), maxiters)


    #hidden layer
    opt = pdealg.opt
    u0 = pdealg.u0
    σᵀ∇u = pdealg.σᵀ∇u
    p1,_re1 = Flux.destructure(u0)
    p2,_re2 = Flux.destructure(σᵀ∇u)
    p3 = [p1;p2;p]
    ps = Flux.params(p3)

    re1 = p -> _re1(p[1:length(p1)])
    re2 = p -> _re2(p[(length(p1)+1):(length(p1)+length(p2))])
    re3 = p -> p[(length(p1)+length(p2)+1):end]

    function F(h, p, t)
        u =  h[end]
        X =  h[1:end-1]
        _σᵀ∇u = re2(p)([X;t])
        _p = re3(p)
        _f = -f(X, u, _σᵀ∇u, _p, t)
        vcat(μ(X,_p,t),[_f])
    end

    function G(h, p, t)
        X = h[1:end-1]
        _p = re3(p)
        _σᵀ∇u = re2(p)([X;t])'
        vcat(σ(X,_p,t),_σᵀ∇u)
    end

    function F(h::Tracker.TrackedArray, p, t)
        u =  h[end]
        X =  h[1:end-1].data
        _σᵀ∇u = σᵀ∇u([X;t])
        _f = -f(X, u, _σᵀ∇u, p, t)
        Tracker.collect(vcat(μ(X,p,t),[_f]))
    end

    function G(h::Tracker.TrackedArray, p, t)
        X = h[1:end-1].data
        _σᵀ∇u = σᵀ∇u([X;t])'
        Tracker.collect(vcat(σ(X,p,t),_σᵀ∇u))
    end

    noise = zeros(Float32,d+1,d)
    prob = SDEProblem{false}(F, G, [X0;0f0], tspan, p3, noise_rate_prototype=noise)

    function neural_sde(init_cond)
        map(1:trajectories) do j #TODO add Ensemble Simulation
            predict_ans = Array(solve(prob, alg, p3;
                                      u0 = init_cond,
                                      p = p3,
                                      save_everystep=false,
                                      sensealg=TrackerAdjoint(),kwargs...))[:,end]
            (X,u) = (predict_ans[1:(end-1)], predict_ans[end])
        end
    end

    function predict_n_sde()
        _u0 = re1(p3)(X0)
        init_cond = [X0;_u0]
        neural_sde(init_cond)
    end

    function loss_n_sde()
        mean(sum(abs2, g(X) - u) for (X,u) in predict_n_sde())
    end

    iters = eltype(X0)[]

    cb = function ()
        save_everystep && push!(iters, u0(X0)[1])
        l = loss_n_sde()
        verbose && println("Current loss is: $l")
        l < pabstol && Flux.stop()
    end

    Flux.train!(loss_n_sde, ps, data, opt; cb = cb)

    save_everystep ? iters : re1(p3)(X0)[1]
end #pde_solve_ns
