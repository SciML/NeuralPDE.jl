struct NNKolmogorov{C,O,S} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    sdealg::S
end
NNKolmogorov(chain  ; opt=Flux.ADAM(0.1) , sdealg = EM()) = NNKolmogorov(chain , opt , sdealg)

function DiffEqBase.solve(
    prob::Union{KolmogorovPDEProblem,SDEProblem},
    alg::NeuralNetDiffEqAlgorithm;
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    save_everystep = false,
    dt,
    kwargs...
    )

    tspan = prob.tspan
    sigma = prob.g
    μ = prob.f
    if prob isa SDEProblem
        xspan = prob.kwargs.data.xspan
        d = prob.kwargs.data.d
        u0 = prob.u0
        phi(xi) = pdf.(u0 , xi)
    else
        xspan = prob.xspan
        d = prob.d
        phi = prob.phi
    end
    ts = tspan[1]:dt:tspan[2]
    xs = xspan[1]:0.001:xspan[2]
    N = size(ts)
    T = tspan[2]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    sdealg = alg.sdealg
    ps     = Flux.params(chain)
    xi     = rand(xs ,d ,N[1])
    #Finding Solution to the SDE having initial condition xi. Y = Phi(S(X , T))
    sdeproblem = SDEProblem(μ,sigma,xi,tspan)
    sol = solve(sdeproblem, sdealg ,dt=0.01 , save_everystep=false , kwargs...)
    x_sde = sol[end]
    y = phi(x_sde)
    data   = Iterators.repeated((xi , y), maxiters)

    #MSE Loss Function
    loss(x , y) =Flux.mse(chain(x), y)

    cb = function ()
        l = loss(xi, y)
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = cb)
    xi , chain(xi)
 end #solve
