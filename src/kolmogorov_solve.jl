struct NNKolmogorov{C,O} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
   
end
NNKolmogorov(chain  ; opt=Flux.ADAM(0.1)) = NNKolmogorov(chain , opt)
 
function DiffEqBase.solve(
    prob::KolmogorovPDEProblem,
    alg::NNKolmogorov;
    abstol = 1f-6,
    verbose = false,
    maxiters = 300,
    save_everystep = false,
    dt,
    )

    tspan = prob.tspan
    phi = prob.phi
    xspan = prob.xspan
    d = prob.d
    T = tspan[2]
    #hidden layer
    chain  = alg.chain
    opt    = alg.opt

    ps     = Flux.params(chain)
    xi     = rand(Uniform(xspan[1] , xspan[2]), d , maxiters )
    N = Normal(0 , sqrt(2. *T ))
    x_sde = pdf.(N , xi)
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
    chain(xi)
 end #solve
