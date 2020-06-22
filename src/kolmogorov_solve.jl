struct NNKolmogorov{C,O,S,E} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    sdealg::S
    ensemblealg::E
end
NNKolmogorov(chain  ; opt=Flux.ADAM(0.1) , sdealg = EM() , ensemblealg = EnsembleThreads()) = NNKolmogorov(chain , opt , sdealg , ensemblealg)

function DiffEqBase.solve(
    prob::Union{KolmogorovPDEProblem,SDEProblem},
    alg::NNKolmogorov;
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    trajectories = 1000,
    save_everystep = false,
    use_gpu = false,
    dt,
    dx,
    kwargs...
    )

    tspan = prob.tspan
    sigma = prob.g
    μ = prob.f
    noise_rate_prototype = prob.noise_rate_prototype
    if prob isa SDEProblem
        xspan = prob.kwargs.data.xspan
        d = prob.kwargs.data.d
        u0 = prob.u0
        phi(xi) = pdf(u0 , xi)
    else
        xspan = prob.xspan
        d = prob.d
        phi = prob.phi
    end
    ts = tspan[1]:dt:tspan[2]
    xs = xspan[1]:dx:xspan[2]
    N = size(ts)
    T = tspan[2]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    sdealg = alg.sdealg
    ensemblealg = alg.ensemblealg
    ps     = Flux.params(chain)
    xi     = rand(xs ,d ,trajectories)
    #Finding Solution to the SDE having initial condition xi. Y = Phi(S(X , T))
    sdeproblem = SDEProblem(μ,sigma,xi,tspan,noise_rate_prototype = noise_rate_prototype)
    function prob_func(prob,i,repeat)
      SDEProblem(prob.f , prob.g , xi[: , i] , prob.tspan ,noise_rate_prototype = prob.noise_rate_prototype)
    end
    output_func(sol,i) = (sol[end],false)
    ensembleprob = EnsembleProblem(sdeproblem , prob_func = prob_func , output_func = output_func)
    sim = solve(ensembleprob, sdealg, ensemblealg , dt=dt, trajectories=trajectories,adaptive=false)
    x_sde  = reshape([],d,0)
    # sol = solve(sdeproblem, sdealg ,dt=0.01 , save_everystep=false , kwargs...)
    # x_sde = sol[end]
    for u in sim.u
        x_sde = hcat(x_sde , u)
    end
    y = phi(x_sde)
    if use_gpu == true
        y = y |>gpu
        xi = xi |> gpu
    end
    data   = Iterators.repeated((xi , y), maxiters)
    if use_gpu == true
        data = data |>gpu
    end

    #MSE Loss Function
    loss(x , y) =Flux.mse(chain(x), y)

    cb = function ()
        l = loss(xi, y)
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = cb)
    chainout = chain(xi)
    xi , chainout
 end #solve
