"""
   PINOODE(chain,
    OptimizationOptimisers.Adam(0.1),
    train_set
    init_params = nothing;
    kwargs...)

## Positional Arguments

* `chain`: A neural network architecture, defined as either a `Flux.Chain` or a `Lux.AbstractExplicitLayer`.
* `opt`: The optimizer to train the neural network.
* `train_set`:
* `init_params`: The initial parameter of the neural network. By default, this is `nothing`
  which thus uses the random initialization provided by the neural network library.

## Keyword Arguments
* `minibatch`: TODO

## Examples

TODO
```julia

```

## References
Zongyi Li "Physics-Informed Neural Operator for Learning Partial Differential Equations"
"""

struct TRAINSET{}
    input_data::Any
    output_data::Any
    u0::Bool
end

function TRAINSET(input_data, output_data; u0 = false)
    TRAINSET(input_data, output_data, u0)
end

struct PINOODE{C, O, P, K} <: DiffEqBase.AbstractODEAlgorithm
    chain::C
    opt::O
    train_set::TRAINSET
    init_params::P
    minibatch::Int
    kwargs::K
end

function PINOODE(chain,
        opt,
        train_set,
        init_params = nothing;
        minibatch = 0,
        kwargs...)
    !(chain isa Lux.AbstractExplicitLayer) && (chain = Lux.transform(chain))
    PINOODE(chain, opt, train_set, init_params, minibatch, kwargs)
end

"""
    PINOPhi(chain::Lux.AbstractExplicitLayer, t, st)
    TODO
"""
mutable struct PINOPhi{C, T, U, S}
    chain::C
    t0::T
    u0::U
    st::S
    function PINOPhi(chain::Lux.AbstractExplicitLayer, t0, u0, st)
        new{typeof(chain), typeof(t0), typeof(u0), typeof(st)}(chain, t0, u0, st)
    end
end

function generate_pino_phi_θ(chain::Lux.AbstractExplicitLayer,
        t0,
        u0,
        init_params)
    θ, st = Lux.setup(Random.default_rng(), chain)
    if init_params === nothing
        init_params = ComponentArrays.ComponentArray(θ)
    else
        init_params = ComponentArrays.ComponentArray(init_params)
    end
    PINOPhi(chain, t0, u0, st), init_params
end

function (f::PINOPhi{C, T, U})(t::Number, θ) where {C <: Lux.AbstractExplicitLayer, T, U}
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), [t]), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    first(y)
end

function (f::PINOPhi{C, T, U})(t::AbstractArray,
        θ) where {C <: Lux.AbstractExplicitLayer, T, U}
    # Batch via data as row vectors
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), t), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    # f.u0 .+ (t[[1], :, :] .- f.t0) .* y
    y
end

function dfdx(phi::PINOPhi, t::AbstractArray, θ)
    ε = [sqrt(eps(eltype(t))), zero(eltype(t))]
    (phi(t .+ ε, θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
end

function inner_physics_loss(phi::PINOPhi{C, T, U}, f, θ, in_::AbstractArray) where {C, T, U}
    ts = in_[[1], :, :] #TODO remove dependence on dimension
    ps = in_[[2], :, :]
    out_ = phi(in_, θ)
    dudt = dfdx(phi, in_, θ)
    fs = f.(out_, ps, ts)
    dudt - fs
end

function inner_physics_loss(phi::PINOPhi{C, T, U},
        f,
        θ,
        in_::AbstractArray,
        p::AbstractArray) where {C, T, U}
    ts = in_[[1], :, :]
    out_ = phi(in_, θ)
    dudt = dfdx(phi, in_, θ)
    fs = f.(out_, p, ts)
    dudt - fs
end

function physics_loss(phi::PINOPhi{C, T, U}, f, θ, train_set::TRAINSET, p) where {C, T, U}
    input_set = train_set.input_data
    data_set_size = size(input_set)[1] * size(input_set[1])[2]
    if train_set.u0 == false
        loss = reduce(vcat,
            [inner_physics_loss(phi, f, θ, in_) for in_ in input_set])
    else #train_set.u!==nothing
        p = fill(p, 1, size(input_set[1])[2], 1)
        loss = reduce(vcat,
            [inner_physics_loss(phi, f, θ, in_, p) for in_ in input_set])
    end
    sum(abs2, loss) / data_set_size
end

function inner_data_loss(phi::PINOPhi{C, T, U},
        θ,
        in_::AbstractArray,
        out_::AbstractArray) where {C, T, U}
    phi(in_, θ) - out_
end

function data_loss(phi::PINOPhi{C, T, U},
        θ,
        train_set::TRAINSET) where {C, T, U}
    input_set, output_set = train_set.input_data, train_set.output_data
    data_set_size = size(input_set)[1] * size(input_set[1])[2]
    loss = reduce(vcat,
        [inner_data_loss(phi, θ, in_, out_) for (in_, out_) in zip(input_set, output_set)])
    sum(abs2, loss) / data_set_size
end

function generate_loss(phi::PINOPhi{C, T, U},
        f,
        train_set::TRAINSET, p) where {C, T, U}
    function loss(θ, _)
        data_loss(phi, θ, train_set) + physics_loss(phi, f, θ, train_set, p)
    end
    return loss
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem,
        alg::PINOODE,
        args...;
        # dt = nothing,
        abstol = 1.0f-6,
        reltol = 1.0f-3,
        verbose = false,
        saveat = nothing,
        maxiters = nothing)
    tspan = prob.tspan
    t0 = tspan[1]
    f = prob.f
    p = prob.p
    u0 = prob.u0
    # param_estim = alg.param_estim

    chain = alg.chain
    opt = alg.opt
    init_params = alg.init_params

    # mapping between functional space of some vararible 'a' of equation (for example initial
    # condition {u(t0 x)} or parameter p) join  and solution of equation u(t)
    train_set = alg.train_set

    !(chain isa Lux.AbstractExplicitLayer) &&
        error("Only Lux.AbstractExplicitLayer neural networks are supported")

    phi, init_params = generate_pino_phi_θ(chain, t0, u0, init_params)

    init_params = ComponentArrays.ComponentArray(init_params)

    isinplace(prob) &&
        throw(error("The PINOODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))

    try
        phi(first(train_set.input_data), init_params) #TODO first(train_set.input_data)
    catch err
        if isa(err, DimensionMismatch)
            throw(DimensionMismatch("Dimensions of the initial u0 and chain should match"))
        else
            throw(err)
        end
    end


    inner_f = generate_loss(phi, f, train_set, p)

    # Creates OptimizationFunction Object from total_loss
    total_loss(θ, _) = inner_f(θ, phi)

    # Optimization Algo for Training Strategies
    opt_algo = Optimization.AutoZygote()

    # Creates OptimizationFunction Object from total_loss
    optf = OptimizationFunction(total_loss, opt_algo)

    iteration = 0
    callback = function (p, l)
        iteration += 1
        verbose && println("Current loss is: $l, Iteration: $iteration")
        l < abstol
    end

    optprob = OptimizationProblem(optf, init_params)
    res = solve(optprob, opt; callback, maxiters, alg.kwargs...)

    # PINOsolution(fullsolution)
    (res, phi)
end
