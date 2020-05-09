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

    chain  = alg.chain
    opt    = alg.opt
    sdealg = alg.sdealg

    prob = SDEProblem(μ,sigma,u0,tspan)
    ensembleprob = EnsembleProblem(prob)
    sim = solve(ensembleprob, sdealg(), EnsembleThreads(), dt=dt,trajectories=1000,adaptive=false)

    payoff = []
    times = []
    iter = 0
    sump = 0
    for u in sim.u
        X = u.u
        m = chain
        a = []
        function Un(n , X )
            if size(a)[1] >= n
                return a[n]
            else
                if(n == 1)
                      ans =  first(m([X[1]])[1])
                      global a = [ans]
                      return ans
                  else
                      ans =  max(first(m([X[n]])[n]) , n + 1 - N)*(1 - sum(Un(i , X ) for i in 1:n-1))
                      global a = vcat( a , ans)
                      return ans
                  end
              end
        end

        function loss()
          return 1000 - sum(Un(i , X )*g(ts[i] , X[i]) for i in 1 : N)
        end
        dataset = Iterators.repeated(() , 50)
        epoch = 0
        opt = ADAM(0.1)

        cb = function ()
            l = loss()
            global a = []
            global epoch = epoch + 1
            epoch%10 == 0 && println("Current loss is: $l")
        end
        Flux.train!(loss, Flux.params(m), dataset, opt; cb = cb)

        Usum = 0
        ti = 0
        for i in 1:N
              a = []
              Usum = Usum + Un(i , X)
              if Usum >= 1 - Un(i , X)
                ti = i
                break
              end
        end
        price = g(ts[ti] , X[ti])
        global payoff = vcat(payoff , price)
        global times = vcat(times, ti)
        global iter = iter + 1
        println(iter)
        global sump = sump + price
        # println("SUM : $sump")
        # println("TIME : $ti")
    end
    sum(payoff)/size(payoff)[1] , sum(times)/size(times)[1]
 end #solve
