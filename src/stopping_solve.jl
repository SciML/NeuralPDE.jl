struct NNStopping{C,O,S,E} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    sdealg::S
    ensemblealg::E
end
NNStopping(chain  ; opt=Flux.ADAM(0.1) , sdealg = EM() , ensemblealg = EnsembleThreads()) = NNStopping(chain , opt , sdealg , ensemblealg)

function DiffEqBase.solve(
    prob::OptimalStoppingProblem,
    alg::NeuralNetDiffEqAlgorithm;
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    trajectories = 1000,
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
    N = size(ts)[1]
    T = tspan[2]

    m  = alg.chain
    opt    = alg.opt
    sdealg = alg.sdealg
    ensemblealg = alg.ensemblealg
    
    prob = SDEProblem(μ,sigma,u0,tspan)
    ensembleprob = EnsembleProblem(prob)
    sim = solve(ensembleprob, sdealg, ensemblealg, dt=dt,trajectories=trajectories,adaptive=false)
    payoff = []
    times = []
    iter = 0
    sump = 0
    for u in sim.u
        X = u.u
        a = []
        function Un(n , X )
            if size(a)[1] >= n
                return a[n]
            else
                if(n == 1)
                      ans =  first(m(X[1])[1])
                      a = [ans]
                      return ans
                  else
                      ans =  max(first(m(X[n])[n]) , n + 1 - N)*(1 - sum(Un(i , X ) for i in 1:n-1))
                      a = vcat( a , ans)
                      return ans
                  end
              end
        end

        function loss()
          return 1000 - sum(Un(i , X )*g(ts[i] , X[i]) for i in 1 : N)
        end
        dataset = Iterators.repeated(() , maxiters)
        epoch = 0
        opt = ADAM(0.1)
        epoch  = 0
        cb = function ()
            l = loss()
            a = []
            epoch = epoch + 1
            epoch%5 == 0 && println("Current loss is: $l")
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
        payoff = vcat(payoff , price)
        times = vcat(times, ti)
        iter = iter + 1
        println(iter)
        sump = sump + price
        # println("SUM : $sump")
        # println("TIME : $ti")
        Flux.loadparams!(m, map(p -> p .= randn.(), Flux.params(m)))
    end
    sum(payoff)/size(payoff)[1]
 end #solve
