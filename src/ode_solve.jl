"""
```julia
NNODE(chain, opt=OptimizationPolyalgorithms.PolyOpt(), init_params = nothing;
                          autodiff=false, batch=0, kwargs...)
```

Algorithm for solving ordinary differential equations using a neural network. This is a specialization
of the physics-informed neural network which is used as a solver for a standard `ODEProblem`.

!!! warn

    Note that NNODE only supports ODEs which are written in the out-of-place form, i.e.
    `du = f(u,p,t)`, and not `f(du,u,p,t)`. If not declared out-of-place then the NNODE
    will exit with an error.

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
* `batch`: The batch size to use for the internal quadrature. Defaults to `0`, which
  means the application of the neural network is done at individual time points one
  at a time. `batch>0` means the neural network is applied at a row vector of values
  `t` simultaniously, i.e. it's the batch size for the neural network evaluations.
  This requires a neural network compatible with batched data.
* `strategy`: The training strategy used to choose the points for the evaluations.
  Default of `nothing` means that `QuadratureTraining` with QuadGK is used if no
  `dt` is given, and `GridTraining` is used with `dt` if given.
* `kwargs`: Extra keyword arguments are splatted to the Optimization.jl `solve` call.

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

## Solution Notes

Note that the solution is evaluated at fixed time points according to standard output handlers
such as `saveat` and `dt`. However, the neural network is a fully continuous solution so `sol(t)`
is an accuate interpolation (up to the neural network training result). In addition, the
`OptimizationSolution` is returned as `sol.k` for further analysis.

## References

Lagaris, Isaac E., Aristidis Likas, and Dimitrios I. Fotiadis. "Artificial neural networks for solving 
ordinary and partial differential equations." IEEE Transactions on Neural Networks 9, no. 5 (1998): 987-1000.
"""
struct NNODE{C,O,P,K,S <: Union{Nothing,AbstractTrainingStrategy}} <: NeuralPDEAlgorithm
    chain::C
    opt::O
    initθ::P
    autodiff::Bool
    batch::Int
    strategy::S
    kwargs::K
end
function NNODE(chain, opt, init_params = nothing;
               strategy = nothing,
               autodiff=false, batch = 0, kwargs...)
    NNODE(chain,opt,init_params,autodiff,batch,strategy,kwargs)
end

"""
```julia
ODEPhi(chain::FastChain, t, u0)
```

Internal, used as a constructor used for representing the ODE solution as a 
neural network in a form that respects boundary conditions, i.e. 
`phi(t) = u0 + t*NN(t)`.
"""
struct ODEPhi{C, T, U}
    chain::C
    t0::T
    u0::U

    ODEPhi(chain::FastChain, t::Number, u0) = new{typeof(chain), typeof(t), typeof(u0)}(chain, t, u0)
    function ODEPhi(chain::Flux.Chain, t, u0)
        p,re = Flux.destructure(chain)
        new{typeof(re), typeof(t), typeof(u0)}(re, t, u0)
    end
end

function (f::ODEPhi{C, T, U})(t,θ) where {C<:FastChain, T, U<:Number} 
    f.u0 + (t-f.t0) * first(f.chain(adapt(parameterless_type(θ),[t]),θ))
end

function (f::ODEPhi{C, T, U})(t,θ) where {C<:FastChain, T, U} 
    f.u0 + (t-f.t0) * f.chain(adapt(parameterless_type(θ),[t]),θ)
end

function (f::ODEPhi{C, T, U})(t,θ) where {C<:Optimisers.Restructure, T, U<:Number} 
    f.u0 + (t-f.t0)*first(f.chain(θ)(adapt(parameterless_type(θ),[t])))
end

function (f::ODEPhi{C, T, U})(t,θ) where {C<:Optimisers.Restructure, T, U} 
    f.u0 + (t-f.t0) * f.chain(θ)(adapt(parameterless_type(θ),[t]))
end

"""
Computes u' using either forward-mode automatic differentiation or
numerical differentiation.
"""
function ode_dfdx(phi::ODEPhi,t::Number,θ,autodiff::Bool)
    if autodiff
        ForwardDiff.derivative(t->phi(t,θ),t)
    else
        (phi(t+sqrt(eps(typeof(t))),θ) - phi(t,θ))/sqrt(eps(typeof(t)))
    end
end

"""
Simple L2 inner loss at a time `t` with parameters θ
"""
function inner_loss(phi,f,autodiff::Bool,t,θ,p)
    sum(abs2,ode_dfdx(phi,t,θ,autodiff) - f(phi(t,θ),p,t))
end

"""
Representation of the loss function, paramtric on the training strategy `strategy`
"""
function generate_loss(strategy::QuadratureTraining,phi,f,autodiff::Bool,tspan,p)
    integrand(t::Number,θ) = abs2(inner_loss(phi,f,autodiff,t,θ,p))
    integrand(ts,θ) = [abs2(inner_loss(phi,f,autodiff,t,θ,p)) for t in ts]

    function loss(θ,p)
        intprob = IntegralProblem(integrand, tspan[1], tspan[2], θ)
        sol = solve(intprob,QuadGKJL();abstol=strategy.abstol,reltol=strategy.reltol)
        sol.u
    end

    # Default this to ForwardDiff until Integrals.jl autodiff is sorted out
    OptimizationFunction(loss,Optimization.AutoForwardDiff())
end

function generate_loss(strategy::GridTraining,phi,f,autodiff::Bool,tspan,p)
    ts = tspan[1]:strategy.dx:tspan[2]

    # sum(abs2,inner_loss(t,θ) for t in ts) but Zygote generators are broken
    function loss(θ,p) 
        sum(abs2,[inner_loss(phi,f,autodiff,t,θ,p) for t in ts])
    end
    optf = OptimizationFunction(loss,Optimization.AutoZygote())
end

function generate_loss(strategy::StochasticTraining,phi,f,autodiff::Bool,tspan,p)
    # sum(abs2,inner_loss(t,θ) for t in ts) but Zygote generators are broken
    function loss(θ,p)
        # (tspan[2]-tspan[1])*rand() + tspan[1] gives Uniform(tspan[1],tspan[2])
        sum(abs2,[inner_loss(phi,f,autodiff,(tspan[2]-tspan[1])*rand() + tspan[1],θ,p) for i in 1:strategy.points])
    end
    optf = OptimizationFunction(loss,Optimization.AutoZygote())
end

function generate_loss(strategy::QuasiRandomTraining,phi,f,autodiff::Bool,tspan)
    error("QuasiRandomTraining is not supported by NNODE since it's for high dimensional spaces only. Use StochasticTraining instead.")
end

struct NNODEInterpolation{T<:ODEPhi,T2}
    phi::T
    θ::T2
end
(f::NNODEInterpolation)(t, idxs::Nothing, ::Type{Val{0}}, p, continuity) = f.phi(t,f.θ)
(f::NNODEInterpolation)(t, idxs, ::Type{Val{0}}, p, continuity) = f.phi(t,f.θ)[idxs]

SciMLBase.interp_summary(::NNODEInterpolation) = "Trained neural network interpolation"

function DiffEqBase.__solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::NNODE,
    args...;
    dt = nothing,
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    reltol = 1f-3,
    verbose = false,
    saveat = nothing,
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
    initθ = alg.initθ

    if initθ === nothing
        if chain isa FastChain
            initθ = DiffEqFlux.initial_params(chain)
        else
            initθ,re  = Flux.destructure(chain)
        end
    else
        initθ = initθ
    end

    if isinplace(prob)
        throw(error("The NNODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))
    end

    phi = ODEPhi(chain,t0,u0)

    try
        phi(t0 , initθ)
    catch err
        if isa(err , DimensionMismatch)
            throw(DimensionMismatch("Dimensions of the initial u0 and chain should match"))
        else
            throw(err)
        end
    end

    strategy = if alg.strategy === nothing
        if dt !== nothing
            GridTraining(dt)
        else
            QuadratureTraining(;quadrature_alg=QuadGKJL(), reltol = convert(eltype(u0),reltol), 
                                abstol = convert(eltype(u0),abstol), maxiters = maxiters, batch=alg.batch)
        end
    else
        alg.strategy
    end

    optf = generate_loss(strategy,phi,f,autodiff::Bool,tspan,p)

    callback = function (p,l)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    
    optprob = OptimizationProblem(optf, initθ)
    res = solve(optprob, opt; callback, maxiters, alg.kwargs...)

    #solutions at timepoints
    if saveat isa Number
        ts = tspan[1]:saveat:tspan[2]
    elseif saveat isa AbstractArray 
        ts = saveat
    elseif dt !== nothing
        ts = tspan[1]:dt:tspan[2]
    elseif save_everystep
        ts = range(tspan[1],tspan[2],length = 100)
    else
        ts = [tspan[1],tspan[2]]
    end

    if u0 isa Number
        u = [first(phi(t,res.u)) for t in ts]
    else
        u = [phi(t,res.u) for t in ts]
    end

    
    sol = DiffEqBase.build_solution(prob,alg,ts,u;
                                    k = res, dense = true,
                                    interp = NNODEInterpolation(phi,res.u), calculate_error = false,
                                    retcode = :Success)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    sol
end #solve
