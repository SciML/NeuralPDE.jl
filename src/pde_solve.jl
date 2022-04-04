struct NNPDEHan{C1,C2,O} <: NeuralPDEAlgorithm
    u0::C1
    σᵀ∇u::C2
    opt::O
end
NNPDEHan(u0,σᵀ∇u;opt=Flux.ADAM(0.1)) = NNPDEHan(u0,σᵀ∇u,opt)

function DiffEqBase.solve(
    prob::TerminalPDEProblem,
    alg::NNPDEHan;
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    save_everystep = false,
    dt,
    give_limit = false,
    trajectories,
    sdealg = EM(),
    ensemblealg = EnsembleThreads(),
    trajectories_upper = 1000,
    trajectories_lower = 1000,
    maxiters_upper = 10,
    )

    X0 = prob.X0
    ts = prob.tspan[1]:dt:prob.tspan[2]
    d  = length(X0)
    g,f,μ,σ,p = prob.g,prob.f,prob.μ,prob.σ,prob.p

    data = Iterators.repeated((), maxiters)


    #hidden layer
    opt = alg.opt
    u0 = alg.u0
    σᵀ∇u = alg.σᵀ∇u
    ps = Flux.params(u0, σᵀ∇u...)

    function sol()
        map(1:trajectories) do j
            u = u0(X0)[1]
            X = X0
            for i in 1:length(ts)-1
                t = ts[i]
                _σᵀ∇u = σᵀ∇u[i](X)
                dW = sqrt(dt)*randn(d)
                u = u - f(X, u, _σᵀ∇u, p, t)*dt + _σᵀ∇u'*dW
                X  = X .+ μ(X,p,t)*dt .+ σ(X,p,t)*dW
            end
            X,u
        end
    end

    function loss()
        mean(sum(abs2,g(X) - u) for (X,u) in sol())
    end

    iters = eltype(X0)[]

    cb = function ()
        save_everystep && push!(iters, u0(X0)[1])
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = cb)


    if give_limit == false
        save_everystep ? iters : u0(X0)[1]
    else
        A = prob.A
        u_domain = prob.u_domain

        ## UPPER LIMIT
        sdeProb = SDEProblem(μ , σ , X0 , prob.tspan)
        ensembleprob = EnsembleProblem(sdeProb)
        sim = solve(ensembleprob, EM(), ensemblealg, dt=dt, trajectories=trajectories_upper,prob.kwargs...)
        function sol_high()
            map(sim.u) do u
                xsde = u.u
                U = g(xsde[end])
                u = u0(X0)[1]
                for i in length(ts):-1:3
                    t = ts[i]
                    _σᵀ∇u = σᵀ∇u[i-1](xsde[i-1])
                    dW = sqrt(dt)*randn(d)
                    U = U .+ f(xsde[i-1], U, _σᵀ∇u, p, t)*dt .- _σᵀ∇u'*dW
                end
                U
            end
        end

        loss_() = sum(sol_high())/trajectories_upper

        ps = Flux.params(u0, σᵀ∇u...)
        cb = function ()
            l = loss_()
            verbose && println("Current loss is: $l")
            l < abstol && Flux.stop()
        end
        dataS = Iterators.repeated((), maxiters_upper)
        Flux.train!(loss_, ps, dataS, ADAM(0.01); cb = cb)
        u_high = loss_()
        ##Lower Limit

        # Function to precalculate the f values over the domain
        function give_f_matrix(X,urange,σᵀ∇u,p,t)
          map(urange) do u
            f(X,u,σᵀ∇u,p,t)
          end
        end

        #The Legendre transform that uses the precalculated f values.
        function legendre_transform(f_matrix , a , urange)
            le = a.*(collect(urange)) .- f_matrix
            return maximum(le)
        end

        function sol_low()
            map(1:trajectories_lower) do j
                u = u0(X0)[1]
                X = X0
                I = zero(eltype(u))
                Q = zero(eltype(u))
                for i in 1:length(ts)-1
                    t = ts[i]
                    _σᵀ∇u = σᵀ∇u[i](X)
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
        save_everystep ? iters : u0(X0)[1] , u_low , u_high
    end

end #pde_solve
