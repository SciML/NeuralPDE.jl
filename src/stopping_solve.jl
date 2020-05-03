struct NNStopping{C,O,S} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    sdealg::S
end
NNStopping(chain  ; opt=Flux.ADAM(0.1) , sdealg = EM()) = NNStopping(chain , opt , sdealg)

function DiffEqBase.solve(
    prob::OptimalStoppingProblem
    alg::NeuralNetDiffEqAlgorithm;
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    save_everystep = false,
    dt,
    kwargs...
    )

    tspan = prob.tspan
    sigma = prob.sigma
    μ = prob.mu
    g = prob.g
    u0 = prob.u0
    ts = tspan[1]:dt:tspan[2]
    N = size(ts)
    T = tspan[2]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    sdealg = alg.sdealg
    ps     = Flux.params(chain)
    ##Temporal Discretisation
    sdeproblem = SDEProblem(μ,sigma,u0,tspan)
    sol = solve(sdeproblem, sdealg ,dt=dt , save_everystep=true , kwargs...)
    X = sol.u

    ##Using the factorisation lemma
    function Un(n , X)
        if(n == 1)return chain(X[1])[1]
        else
            return max(first(chain(X[n])[n]) , n + 1 - N)*(1 - sum(Un(i , X) for i in 1:n-1))
        end
    end

    data   = Iterators.repeated((), maxiters)

    function loss()
        return sum(Un(i , X)*g(ts[i] , X[i]) for i in 1 : N) / N
    end

    cb = function ()
        l = loss(xi, y)
        verbose && println("Current loss is: $l")
        # l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = cb)

 end #solve
