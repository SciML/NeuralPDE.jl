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
      if(n == 1)return m(X[1])[1]
      else
          return max(first(m(X[n])[n]) , n + 1 - N)*(1 - sum(Un(i , X) for i in 1:n-1))
      end
    end

    function loss()
        return sum(Un(i , X)*g(ts[i] , X[i]) for i in 1 : N)
    end

    dataset = Iterators.repeated(() , maxiters)
    epoch = 0
    opt = ADAM(-0.1)
    cb = function ()
      l = loss()
      epoch = epoch + 1
      epoch%20 == 0 && println("Current loss is: $l")
    end
    Flux.train!(loss, Flux.params(m), dataset, opt; cb = cb)
    Usum = 0
    time = 0
    for i in 1:N
        global Usum = Usum + Un(i , X)
        if Usum >= 1 - Un(i , X)
            time = i
        break
    end
  end
  g(ts[time] , X[time]) , time
 end #solve
