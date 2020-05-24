struct NNStopping{C,O,S,E} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    sdealg::S
    ensemblealg::E
end
NNStopping(chain  ; opt=Flux.ADAM(0.1) , sdealg = EM() , ensemblealg = EnsembleThreads()) = NNStopping(chain , opt , sdealg , ensemblealg)

function DiffEqBase.solve(
    prob::SDEProblem,
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
    sigma = prob.g
    μ = prob.f
    g = prob.kwargs.data.g
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
    # for u in sim.u
    un = []
    function Un(n , X )
        if size(un)[1] >= n
            return un[n]
        else
            if(n == 1)
                  ans =  first(m(X[1])[1])
                  un = [ans]
                  return ans
              else
                  ans =  max(first(m(X[n])[n]) , n + 1 - size(ts)[1])*(1 - sum(Un(i , X ) for i in 1:n-1))
                  un = vcat( un , ans)
                  return ans
              end
          end
    end

    function loss()
        reward = 0.00
        for u in sim.u
            X = u.u
            reward = reward + sum(Un(i , X )*g(ts[i] , X[i]) for i in 1 : size(ts)[1])
            un = []
        end
        return 10000 - reward
    end
    dataset = Iterators.repeated(() , maxiters)
    epoch = 0
    epoch  = 0
    cb = function ()
        l = loss()
        un = []
        epoch = epoch + 1
        epoch%5 == 0 && println("Current loss is: $l")
    end
    Flux.train!(loss, Flux.params(m), dataset, opt; cb = cb)

    Usum = 0
    ti = 0
    Xt = sim.u[1].u
    for i in 1:N
          un = []
          Usum = Usum + Un(i , Xt)
          if Usum >= 1 - Un(i , Xt)
            ti = i
            break
          end
    end
    for u in sim.u
        X = u.u
        price = g(ts[ti] , X[ti])
        payoff = vcat(payoff , price)
        times = vcat(times, ti)
        iter = iter + 1
        # println("SUM : $sump")
        println("TIME : $ti")
    end
    sum(payoff)/size(payoff)[1]
 end #solve
