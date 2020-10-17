"""
Algorithm for solving Optimal Stopping Problems.

```julia
NeuralPDE.NNStopping(chain, opt, sdealg, ensemblealg )
```
Arguments:
- `chain`: A Chain neural network with an N-dimensional output according to N stopping times and the last layer (softmax function).
- `opt`: The optimizer to train the neural network. Defaults to `ADAM(0.1)`.
- `sdealg`: The algorithm used to solve the discretized SDE according to the process that X follows. Defaults to `EM()`.
- `ensemblealg`: The algorithm used to solve the Ensemble Problem that performs Ensemble simulations for the SDE. Defaults to `EnsembleThreads()`. See
  the [Ensemble Algorithms](https://diffeq.sciml.ai/stable/features/ensemble/#EnsembleAlgorithms-1)
  documentation for more details.

[1]Becker, Sebastian, et al. "Solving high-dimensional optimal stopping problems using deep learning." arXiv preprint arXiv:1908.01602 (2019).
"""
struct NNStopping{C,O,S,E} <: NeuralPDEAlgorithm
    chain::C
    opt::O
    sdealg::S
    ensemblealg::E
end
NNStopping(chain  ; opt=Flux.ADAM(0.1) , sdealg = EM() , ensemblealg = EnsembleThreads()) = NNStopping(chain , opt , sdealg , ensemblealg)

function DiffEqBase.solve(
    prob::SDEProblem,
    alg::NNStopping;
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

    cb = function ()
        l = loss()
        un = []
        println("Current loss is: $l")
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
        # println("TIME : $ti")
    end
    sum(payoff)/size(payoff)[1]
 end #solve
