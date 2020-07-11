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
    dt ,
    pabstol = 1f-6,
    save_everystep = false,
    give_limit = false,
    ensemblealg = EnsembleThreads(),
    trajectories_upper = 1000,
    trajectories_lower = 1000,
    maxiters_upper = 10,
    kwargs...)

    X0 = prob.X0
    x0 = prob.X0
    tspan = prob.tspan
    d  = length(X0)
    kwargs = prob.kwargs
    g,f,μ,σ = prob.g,prob.f,prob.μ,prob.σ
    p = prob.p isa AbstractArray ? prob.p : Float32[]
    A = prob.A
    u_domain = prob.u_domain
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
            predict_ans = Array(concrete_solve(prob, alg, dt=dt,init_cond, p3;
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


    if give_limit == false
        save_everystep ? iters : re1(p3)(X0)[1]
    else
    ## UPPER LIMIT
        sdeProb = SDEProblem(μ , σ , x0 , tspan , noise_rate_prototype = zeros(Float32,d,d))
        output_func(sol,i) = (sol[end],false)
        ensembleprob = EnsembleProblem(sdeProb , output_func = output_func)
        sim_f = solve(ensembleprob, alg, ensemblealg, dt=dt, trajectories = trajectories_upper )
        Xn = reduce( vcat ,sim_f.u )
        Un = collect(g(X) for X in Xn)

        tspan_rev = (tspan[2] , tspan[1])
        sdeProb2 = SDEProblem(F, G, [Xn[1] ; Un[1]] ,tspan_rev, p3 ,  noise_rate_prototype=noise)
        function prob_func(prob,i,repeat)
          SDEProblem(prob.f , prob.g , [Xn[i] ; Un[i]] , prob.tspan , prob.p ,noise_rate_prototype = prob.noise_rate_prototype)
        end

        ensembleprob2 = EnsembleProblem(sdeProb2 , prob_func = prob_func  , output_func   = output_func)
        sim = solve(ensembleprob2, alg, ensemblealg, dt=dt, trajectories=trajectories_upper, output_func = output_func,save_everystep = false ,sensealg=TrackerAdjoint())

        function sol_high()
            map(sim.u) do u
                u[2]
            end
        end

        loss_() = sum(sol_high())/trajectories_upper

        ps = Flux.params(u0, σᵀ∇u...)
        cb = function ()
            l = loss_()
            true && println("Current loss is: $l")
            l < 1e-6 && Flux.stop()
        end
        dataS = Iterators.repeated((), maxiters_upper)
        Flux.train!(loss_, ps, dataS, ADAM(0.01); cb = cb)
        u_high = loss_()
        # Function to precalculate the f values over the domain
        function give_f_matrix(X,urange,σᵀ∇u,p,t)
          map(urange) do u
            f(X,u,σᵀ∇u,p,t)
          end
        end

        #The legendre transform that uses the precalculated f values.
        function legendre_transform(f_matrix , a , urange)
            le = a.*(collect(urange)) .- f_matrix
            return maximum(le)
        end
        ts = tspan[1]:dt:tspan[2]
        function sol_low()
            map(1:trajectories_lower) do j
                u = u0(X0)[1]
                X = X0
                I = zero(eltype(u))
                Q = zero(eltype(u))
                for i in 1:length(ts)-1
                    t = ts[i]
                    _σᵀ∇u = σᵀ∇u([X ; 0.0f0])
                    dW = sqrt(dt)*randn(d)
                    u = u - f(X, u, _σᵀ∇u, p, t)*dt + _σᵀ∇u'*dW
                    X  = X .+ μ(X,p,t)*dt .+ σ(X,p,t)*dW
                    f_matrix = give_f_matrix(X , u_domain, _σᵀ∇u, p, ts[i])
                    a_ = A[findmax(collect(A).*u .- collect(legendre_transform(f_matrix, a, u_domain) for a in A))[2]]
                    I = I + a_*dt
                    Q = Q + exp(I)*legendre_transform(f_matrix, a_, u_domain)
                end
                I , Q , X
            end
        end
        u_low = sum(exp(I)*g(X) - Q for (I ,Q ,X) in sol_low())/(trajectories_lower)
        save_everystep ? iters : re1(p3)(X0)[1] , u_low , u_high
    end
end #pde_solve_ns
