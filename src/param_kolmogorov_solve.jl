struct NNParamKolmogorov{C,O,S,E} <: NeuralPDEAlgorithm
    chain::C
    opt::O
    sdealg::S
    ensemblealg::E
end
NNParamKolmogorov(chain  ; opt=Flux.ADAM(0.1) , sdealg = EM() , ensemblealg = EnsembleThreads()) = NNParamKolmogorov(chain , opt , sdealg , ensemblealg)

function DiffEqBase.solve(
    prob::ParamKolmogorovPDEProblem,
    alg::NNKolmogorov;
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    trajectories = 1000,
    save_everystep = false,
    use_gpu = false,
    dγ = 0.01,
    dt,
    dx,
    kwargs...
    )

    tspan = prob.tspan
    g = prob.g
    f = prob.f
    noise_rate_prototype = prob.noise_rate_prototype
    yspan = prob.yspan
    xspan = prob.xspan
    d = prob.d
    phi = prob.phi
    ts = tspan[1]:dt:tspan[2]
    xs = xspan[1]:dx:xspan[2]
    γs = Y_domain[1]:dγ:Y_domain[2]

    γ_mu_prototype = prob.γ_mu_prototype
    γ_sigma_prototype = prob.γ_sigma_prototype
    γ_phi_prototype = prob.γ_phi_prototype

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    sdealg = alg.sdealg
    ensemblealg = alg.ensemblealg
    ps     = Flux.params(chain)
    γ_sigma = rand(γs , d , d , d + 1, trajectories)
    γ_phi = rand(γs , k , 1 , 1, trajectories)
    γ_mu_1 = rand(γs , d , d , 1, trajectories)
    γ_mu_2 = rand(γs , d , 1 , 1, trajectories)
    t = rand(ts , 1 , trajectories)
    x = rand(xs , d , 1 , trajectories)

    #Sample all the parameters from uniform distributions
    total_dims = d + 1
    if isnothing(γ_sigma_prototype)
      γ_sigma = nothing
    else
      γ_sigma = rand(y_domain[1]:dy:y_domain[2] , size(γ_sigma_prototype)[1] , size(γ_sigma_prototype)[2] , size(γ_sigma_prototype)[3] , trajectories)
      total_dims += d^2*(size(γ_sigma_prototype))[3]
    end
    if isnothing(γ_mu_prototype)
      γ_mu_1 = nothing
      γ_mu_2 = nothing
    else
      if !isnothing(γ_mu_prototype[1])
        γ_mu_1 = rand(y_domain[1]:dy:y_domain[2] , size(γ_mu_prototype[1])[1], size(γ_mu_prototype[1])[2], 1 , trajectories)
        total_dims += d^2
      elseif !isnothing(γ_mu_prototype[1])
        γ_mu_2 = rand(y_domain[1]:dy:y_domain[2] , size(γ_mu_prototype[2])[1], size(γ_mu_prototype[2])[2] , 1 ,trajectories)
        total_dims += d
      end
    end
    if isnothing(γ_phi_prototype)
      γ_phi = nothing
    else
      γ_phi = rand(0.00:0.01:0.00 , size(γ_phi_prototype)[1] , size(γ_phi_prototype)[2] , 1 , trajectories)
      total_dims += size(γ_phi_prototype)[1]
    end

    #Declare wrapper functions to mu and sigma
    function sigma_(x , γ_sigma , i)
      if isnothing(γ_sigma)
        return g(x , γ_sigma)
      else
        return g(x , γ_sigma[: , : , : , i])
      end
    end

    function mu_(x , γ_mu_1, γ_mu_2 , i)
      if isnothing(γ_mu_prototype)
        return f(x , γ_mu_1 , γ_mu_2)
      else
        if isnothing(γ_mu_1) && isnothing(γ_mu_2)
          return f(x , γ_mu_1 , γ_mu_2)
        elseif isnothing(γ_mu_1) && !isnothing(γ_mu_2)
          return f(x , γ_mu_1 , γ_mu_2[: , : , : , i])
        elseif !isnothing(γ_mu_1) && isnothing(γ_mu_2)
          return f(x , γ_mu_1[: , : , : , i] , γ_mu_2)
        else
          return f(x , γ_mu_1[: , : , : , i] , γ_mu_2[: , : , : , i])
        end
      end
    end


    #Preparing training data
    X  = reshape([], total_dims ,0)
    for i in 1:length(t)
      K = vcat(t[i] , x[: , : , i])
      if !isnothing(γ_sigma_prototype)
        K = vcat(K , reshape(γ_sigma[: , : , : , i] , d^2*(size(γ_sigma_prototype))[3] , 1))
      end
      if !isnothing(γ_mu_prototype)
        K = vcat(K , reshape(γ_mu_1[: , : , : , i] , d^2 , 1) , reshape(γ_mu_2[: , : , : , i]  , d , 1))
      end
      if !isnothing(γ_phi_prototype)
        K = vcat(K , reshape(γ_mu_1[: , : , : , i] , d^2 , 1) , reshape(γ_mu_2[: , : , : , i]  , d , 1))
      end
      X = hcat(X , K)
    end
    print("Data Prepared to train the model")

    function sigma(x , p , t)
      sigma_(x , γ_sigma , 1)
    end
    function mu(x , p , t)
      mu_(x , γ_mu_1, γ_mu_2 , 1)
    end

    sdeproblem = SDEProblem(mu,sigma,x[: , : , 1], (0.00 , t[1]))
    function prob_func(prob,i,repeat)
      sigma(x , p , t) = sigma_(x , γ_sigma , i)
      mu(x , p , t) = mu_(x , γ_mu_1, γ_mu_2 , i)
      SDEProblem(prob.f , prob.g , x[: , : , i] , (0.00 , t[i]) ,noise_rate_prototype = prob.noise_rate_prototype)
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
    y = phi(x_sde , γ_phi)
    y = phi(x_sde)
    if use_gpu == true
        y = y |>gpu
        X = X |> gpu
    end
    data   = Iterators.repeated((X , y), 1000)
    if use_gpu == true
        data = data |>gpu
    end

    #MSE Loss Function
    loss(x , y) =Flux.mse(chain(x) , y)

    cb = function ()
        l = loss(X, y)
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = cb)


 end #solve
