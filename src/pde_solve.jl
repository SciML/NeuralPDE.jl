struct NNPDEHan{C1,C2,O} <: NeuralNetDiffEqAlgorithm
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
    trajectories)

    X0 = prob.X0
    ts = prob.tspan[1]:dt:prob.tspan[2]
    d  = length(X0)
    g,f,μ,σ,p = prob.g,prob.f,prob.μ,prob.σ,prob.p
    A = prob.kwargs.data.A
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
        ## UPPER LIMIT
        sdeProb = SDEProblem(μ , σ , X0 , prob.tspan)
        ensembleprob = EnsembleProblem(sdeProb)
        sim = solve(ensembleprob, EM(), EnsembleThreads(), dt=dt,trajectories=800,adaptive=false)
        function sol_high()
            Uo = []
            p = nothing
            for u in sim.u
                xsde = u.u
                U = g(xsde[end])
                u = u0(x0)[1]
                for i in length(ts)-1:-1:1
                    t = ts[i]
                    _σᵀ∇u = σᵀ∇u[i](xsde[i])
                    dW = sqrt(dt)*randn(d)
                    U = U .+ f(xsde[i], u, _σᵀ∇u, p, t)*dt .- _σᵀ∇u'*dW
                end
                Uo = vcat(Uo , U)
            end
            Uo
        end
        loss_() = sum(sol_high())/800
        data = Iterators.repeated((), 20)
        Flux.train!(loss, ps, dataset, opt; cb = cb)
        u_high = loss_()
        println(u_high)

        ##Lower Limit
        u_domain = -100:0.1:100
        println(u0(X0)[1])
        function give_f_matrix(X,urange,σᵀ∇u,p,t)
            a = []
            for u in urange
                a = vcat(a , f(X,u,σᵀ∇u,p,t))
            end
            return a
        end

        function legendre_transform(f_matrix , a , urange)
            le = a.*(collect(urange)) .- f_matrix
            return maximum(le)
        end

        m2 = 1000
        function sol_low()
            p = nothing
            map(1:m2) do j
                u = u0(x0)[1]
                X = x0
                I = 0.0
                Q = 0.0
                for i in 1:length(ts)-1
                    t = ts[i]
                    _σᵀ∇u = σᵀ∇u[i](X)
                    dW = sqrt(dt)*randn(d)
                    u = u - f(X, u, _σᵀ∇u, p, t)*dt + _σᵀ∇u'*dW
                    X  = X .+ μ_f(X,p,t)*dt .+ σ_f(X,p,t)*dW
                    f_matrix = give_f_matrix(X , u_domain, _σᵀ∇u, nothing, ts[i])
                    a_ = A[findmax(collect(A).*u .- collect(legendre_transform(f_matrix, a, u_domain) for a in A))[2]]
                    I = I + a_*dt
                    Q = Q + exp(I)*legendre_transform(f_matrix, a_, u_domain)
                end
                I , Q ,X
            end
        end
        u_low = sum(exp(I)*g(X) - Q for (I ,Q , X) in sol_low())/(m2)
        println(u_low)
        save_everystep ? iters : u0(X0)[1] , u_low , u_high
    end

end #pde_solve
