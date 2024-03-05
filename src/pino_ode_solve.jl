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
#TODO
struct TRAINSET{} #T
    input_data::Vector{ODEProblem}
    output_data::Array
    isu0::Bool
end

function TRAINSET(input_data, output_data; isu0 = false)
    TRAINSET(input_data, output_data, isu0)
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
    #TODO transform convert complex numbers to zero
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
        new{typeof(chain), typeof(t0), typeof(u0), typeof(st)}(chain, t0,u0, st)
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

# function (f::PINOPhi{C, T, U})(t::Number, θ) where {C <: Lux.AbstractExplicitLayer, T, U}
#     y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), [t]), θ, f.st)
#     ChainRulesCore.@ignore_derivatives f.st = st
#     first(y)
# end

function (f::PINOPhi{C, T, U})(t::AbstractArray,
        θ) where {C <: Lux.AbstractExplicitLayer, T, U}
    # Batch via data as row vectors
    # y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), t), θ, f.st)
    y, st = f.chain(t, θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    # y
    f.u0 .+ (t[[1],:,:] .- f.t0) .* y
end

#     feature_dims = 2:(ndims(t) - 1)
# loss = sum( t, dims = feature_dims)
    # loss = sum(.√(sum(abs2, 𝐲̂ - 𝐲, dims = feature_dims)))
    # y_norm = sum(.√(sum(abs2, 𝐲, dims = feature_dims)))

    # return loss / y_norm
# function dfdx(phi::PINOPhi, t::AbstractArray, θ)
#     ε = sqrt(eps(eltype(t)))
#     εs = [ε, zero(eltype(t))]
#     # ε = [sqrt(eps(eltype(t))), zeros(eltype(t), phi.chain.layers.layer_1.in_dims - 1)...]
#     # ChainRulesCore.@ignore_derivatives tl = t .+ ε
#     tl = t .+ ε
#     tr = t
#     (phi(tl, θ) - phi(tr, θ)) ./ ε
# end

function dfdx(phi::PINOPhi, t::AbstractArray, θ)
    ε = [sqrt(eps(eltype(t))), zero(eltype(t))]
    #TODO ε is size of t ?
    # ε = [sqrt(eps(eltype(t))), zeros(eltype(t), phi.chain.layers.layer_1.in_dims - 1)...]
    (phi(t .+ ε, θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
end

function inner_physics_loss(phi::PINOPhi{C, T, U},
        θ,
        ts::AbstractArray,
        prob::ODEProblem,
        isu0::Bool,
        in_) where {C, T, U}
    u0 = prob.u0
    p = prob.p
    f = prob.f
    # if isu0 == true
    #     #TODO data should be generate before train ?
    #     in_ = reduce(vcat, [ts, fill(u0, 1, size(ts)[2], 1)])
    #     #TODO for all case p and u0
    #     # u0 isa Vector
    #     # in_ = reduce(vcat, [ts, reduce(hcat, fill(u0, 1, size(ts)[2], 1))])
    # else
    #     if p isa Number
    #         in_ = reduce(vcat, [ts, fill(p, 1, size(ts)[2], 1)])
    #     elseif p isa Vector
    #         #TODO nno for Vector
    #         inner = reduce(vcat, [ts, reduce(hcat, fill(p, 1, size(ts)[2], 1))])
    #         in_ = reshape(inner, size(inner)..., 1)
    #     else
    #         error("p should be a number or a vector")
    #     end
    # end
    out_ = phi(in_, θ)
    # fs = f.f.(out_, p, ts)
    if p isa Number
        fs = f.f.(out_, p, ts)
    elseif p isa Vector
        fs = reduce(hcat, [f.f(out_[:, i], p, ts[i]) for i in 1:size(out_, 2)])
    else
        error("p should be a number or a vector")
    end
    NeuralOperators.l₂loss(dfdx(phi, in_, θ), fs)
end


function physics_loss(phi::PINOPhi{C, T, U},
        θ,
        ts::AbstractArray,
        train_set::TRAINSET,
        input_data_set) where {C, T, U}
    prob_set, output_data = train_set.input_data, train_set.output_data
    f = prob_set[1].f
    # norm = prod(size(output_data))
    # norm = size(output_data)[1] * size(output_data[1])[2] * size(output_data[1])[1]
    # loss = reduce(vcat,
    #     [inner_physics_loss(phi, θ, ts, prob, train_set.isu0, in_)
    #      for (in_, prob) in zip(inputdata, prob_set)])
    # sum(abs2, loss) / norm
    ps = [prob.p for prob in prob_set]'
    fs = f.f.(output_data, ps, ts)
    loss = NeuralOperators.l₂loss(dfdx(phi, input_data_set, θ), fs)
end

function inner_data_loss(phi::PINOPhi{C, T, U},
        θ,
        ts::AbstractArray,
        prob::ODEProblem,
        out_::AbstractArray,
        isu0::Bool,
        in_) where {C, T, U}
    u0 = prob.u0
    p = prob.p
    f = prob.f
    if isu0 == true
        in_ = reduce(vcat, [ts, fill(u0, 1, size(ts)[2], 1)])
        #TODO for all case p and u0
        # u0 isa Vector
        # in_ = reduce(vcat, [ts, reduce(hcat, fill(u0, 1, size(ts)[2], 1))])
    else
        if p isa Number
            in_ = reduce(vcat, [ts, fill(p, 1, size(ts)[2],1)])
        elseif p isa Vector
            inner = reduce(vcat, [ts, reduce(hcat, fill(p, 1, size(ts)[2], 1))])
            in_ = reshape(inner, size(inner)..., 1)
        else
            error("p should be a number or a vector")
        end
    end
    NeuralOperators.l₂loss(phi(in_, θ), out_)
end

function data_loss(phi::PINOPhi{C, T, U},
        θ,
        ts::AbstractArray,
        train_set::TRAINSET,
        input_data_set) where {C, T, U}
    prob_set, output_data = train_set.input_data, train_set.output_data
    # norm = prod(size(output_data))
    # norm = size(output_data)[1] * size(output_data[1])[2] * size(output_data[1])[1]
    # loss = reduce(vcat,
    #     [inner_data_loss(phi, θ, ts, prob, out_, train_set.isu0, in_)
    #      for (prob, out_, in_) in zip(prob_set, output_data, input_data_set)])
    # sum(abs2, loss) / norm
    loss = NeuralOperators.l₂loss(phi(input_data_set, θ), output_data)
end

function generate_data(ts, prob_set, isu0)
    input_data_set = []
    input_data_set_right = []
    for prob in prob_set
        u0 = prob.u0
        p = prob.p
        f = prob.f
        ε = sqrt(eps(eltype(ts)))
        tsr = ts .+ ε
        if isu0 == true
            #TODO data should be generate before train ?
            in_ = reduce(vcat, [ts, fill(u0, 1, size(ts)[2], 1)])

            #TODO for all case p and u0
            # u0 isa Vector
            # in_ = reduce(vcat, [ts, reduce(hcat, fill(u0, 1, size(ts)[2], 1))])
        else
            if p isa Number
                in_ = reduce(vcat, [ts, fill(p, 1, size(ts)[2], 1)])
                in_r = reduce(vcat, [tsr, fill(p, 1, size(ts)[2], 1)])

            elseif p isa Vector
                #TODO nno for Vector
                inner = reduce(vcat, [ts, reduce(hcat, fill(p, 1, size(ts)[2], 1))])
                in_ = reshape(inner, size(inner)..., 1)
            else
                error("p should be a number or a vector")
            end
        end
        push!(input_data_set, in_)
        push!(input_data_set_right, in_r)
    end
    input_data_set, input_data_set_right
end

function generate_data_matrix(ts, prob_set, isu0)

    batch_size = size(prob_set)[1]
    instances_size = size(ts)[2]
    dims = 2
    input_data_set = Array{Float32, 3}(undef, dims, instances_size, batch_size)
    for (i,prob) in enumerate(prob_set)
        u0 = prob.u0
        p = prob.p
        f = prob.f
        if isu0 == true
            #TODO data should be generate before train ?
            in_ = reduce(vcat, [ts, fill(u0, 1, size(ts)[2], 1)])

            #TODO for all case p and u0
            # u0 isa Vector
            # in_ = reduce(vcat, [ts, reduce(hcat, fill(u0, 1, size(ts)[2], 1))])
        else
            if p isa Number
                in_ = reduce(vcat, [ts, fill(p, 1, size(ts)[2], 1)])
            elseif p isa Vector
                #TODO nno for Vector
                inner = reduce(vcat, [ts, reduce(hcat, fill(p, 1, size(ts)[2], 1))])
                in_ = reshape(inner, size(inner)..., 1)
            else
                error("p should be a number or a vector")
            end
        end
        input_data_set[:, :, i] = in_
    end
    input_data_set
end

function generate_loss(phi::PINOPhi{C, T, U}, train_set::TRAINSET, tspan) where {C, T, U}
    t0 = tspan[1]
    t_end = tspan[2]
    instances_size = size(train_set.output_data)[2]
    # instances_size = size(train_set.output_data[1])[2]
    range_ = range(t0, stop = t_end, length = instances_size)
    ts = reshape(collect(range_), 1, instances_size)
    prob_set, output_data = train_set.input_data, train_set.output_data
    input_data_set = generate_data_matrix(ts, prob_set, train_set.isu0)
    function loss(θ, _)
        data_loss(phi, θ, ts, train_set, input_data_set) +
        physics_loss(phi, θ, ts, train_set, input_data_set)
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
    # f = prob.f
    # p = prob.p
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
        # phi(rand(5, 100, 1), init_params) #TODO input data
    catch err
        if isa(err, DimensionMismatch)
            throw(DimensionMismatch("Dimensions of the initial u0 and chain should match"))
        else
            throw(err)
        end
    end

    total_loss = generate_loss(phi, train_set, tspan)

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
