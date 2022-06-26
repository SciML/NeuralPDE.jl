"""
```julia
NNODE(chain, opt=OptimizationPolyalgorithms.PolyOpt(), init_params = nothing;
                          autodiff=false,kwargs...)
```

Algorithm for solving ordinary differential equations using a neural network. This is a specialization
of the physics-informed neural network which is used as a solver for a standard `ODEProblem`.

## Positional Arguments

* `chain`: A neural network architecture, defined as either a `Flux.Chain` or a `Lux.Chain`.
* `opt`: The optimizer to train the neural network. Defaults to `OptimizationPolyalgorithms.PolyOpt()`
* `initθ`: The initial parameter of the neural network. By default this is `nothing`
  which thus uses the random initialization provided by the neural network library.

## Keyword Arguments

* `autodiff`: The switch between automatic and numerical differentiation for
              the PDE operators. The reverse mode of the loss function is always
              automatic differentation (via Zygote), this is only for the derivative
              in the loss function (the derivative with respect to time).

## Example

```julia
f(u,p,t) = cos(2pi*t)
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(linear, u0 ,tspan)
chain = Flux.Chain(Dense(1,5,σ),Dense(5,1))
opt = Flux.ADAM(0.1)
sol = solve(prob, NeuralPDE.NNODE(chain,opt), dt=1/20f0, verbose = true,
            abstol=1e-10, maxiters = 200)
```

## References

Lagaris, Isaac E., Aristidis Likas, and Dimitrios I. Fotiadis. "Artificial neural networks for solving 
ordinary and partial differential equations." IEEE Transactions on Neural Networks 9, no. 5 (1998): 987-1000.
"""
struct NNODE{C,O,P,K} <: NeuralPDEAlgorithm
    chain::C
    opt::O
    initθ::P
    autodiff::Bool
    kwargs::K
end
function NNODE(chain, opt, init_params = nothing; autodiff=false, kwargs...)
    if init_params === nothing
        if chain isa FastChain
            initθ = DiffEqFlux.initial_params(chain)
        else
            initθ,re  = Flux.destructure(chain)
        end
    else
        initθ = init_params
    end
    NNODE(chain,opt,initθ,autodiff,kwargs)
end

function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::NNODE,
    args...;
    dt,
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t0 = tspan[1]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    autodiff = alg.autodiff

    #train points generation
    ts = tspan[1]:dt:tspan[2]
    initθ = alg.initθ

    if isinplace(prob)
        throw(error("The NNODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))
    end
    
    if chain isa FastChain
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> u0 + (t-tspan[1]) * first(chain(adapt(parameterless_type(θ),[t]),θ))
        else
            phi = (t,θ) -> u0 + (t-tspan[1]) * chain(adapt(parameterless_type(θ),[t]),θ)
        end
    else
        _,re  = Flux.destructure(chain)
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> u0 + (t-tspan[1])*first(re(θ)(adapt(parameterless_type(θ),[t])))
        else
            phi = (t,θ) -> u0 + (t-tspan[1]) * re(θ)(adapt(parameterless_type(θ),[t]))
        end
    end

    try
        phi(t0 , initθ)
    catch err
        if isa(err , DimensionMismatch)
            throw(DimensionMismatch("Dimensions of the initial u0 and chain should match"))
        else
            throw(err)
        end
    end

    if autodiff
        dfdx = (t,θ) -> ForwardDiff.derivative(t->phi(t,θ),t)
    else
        dfdx = (t,θ) -> (phi(t+sqrt(eps(typeof(t))),θ) - phi(t,θ))/sqrt(eps(typeof(t)))
    end

    function inner_loss(t,θ)
        sum(abs2,dfdx(t,θ) - f(phi(t,θ),p,t))
    end
    loss(θ) = sum(abs2,[inner_loss(t,θ) for t in ts]) # sum(abs2,inner_loss(t,θ) for t in ts) but Zygote generators are broken

    callback = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = callback, maxiters=maxiters, alg.kwargs...)

    #solutions at timepoints
    if u0 isa Number
        u = [first(phi(t,res.minimizer)) for t in ts]
    else
        u = [phi(t,res.minimizer) for t in ts]
    end

    sol = DiffEqBase.build_solution(prob,alg,ts,u,calculate_error = false)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    sol
end #solve
