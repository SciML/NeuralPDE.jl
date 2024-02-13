struct PINOODE{C, O, T, P, K} <: DiffEqBase.AbstractODEAlgorithm
    chain::C
    opt::O
    training_mapping::T
    init_params::P
    minibatch::Int
    kwargs::K
end

function PINOODE(chain,
        opt,
        training_mapping,
        init_params = nothing;
        minibatch = 0,
        kwargs...)
    !(chain isa Lux.AbstractExplicitLayer) && (chain = Lux.transform(chain))
    PINOODE(chain, opt, training_mapping, init_params, minibatch, kwargs)
end

"""
    PINOPhi(chain::Lux.AbstractExplicitLayer, t, st)
"""
mutable struct PINOPhi{C, S}
    chain::C
    st::S
    function PINOPhi(chain::Lux.AbstractExplicitLayer, st)
        new{typeof(chain), typeof(st)}(chain,  st)
    end
end

function generate_pino_phi_θ(chain::Lux.AbstractExplicitLayer, init_params)
    θ, st = Lux.setup(Random.default_rng(), chain)
    if init_params === nothing
        init_params = ComponentArrays.ComponentArray(θ)
    else
        init_params = ComponentArrays.ComponentArray(init_params)
    end
    PINOPhi(chain, st), init_params
end

function (f::PINOPhi{C})(t::Number, θ) where {C <: Lux.AbstractExplicitLayer}
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), [t]), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    # f.u0 .+ (t .- f.t0) .* y
    y
end

function (f::PINOPhi{C})(t::AbstractArray, θ) where {C <: Lux.AbstractExplicitLayer}
    # Batch via data as row vectors
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), t), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    # f.u0 .+ (t' .- f.t0) .* y
    y
end

function dfdx(phi::PINOPhi, t::AbstractArray, θ)
    ε = [sqrt(eps(eltype(t))), zero(eltype(t))]
    (phi(t .+ ε, θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
end

function inner_physics_loss(phi::PINOPhi{C}, f, θ, in_) where {C}
    out_ = phi(in_, θ)
    dudt = dfdx(phi, in_, θ)
    ts = in_[[1], :]
    ps = in_[[2], :]
    fs = f.(out_, ps, ts)
    dudt - fs
end

function physics_loss(phi::PINOPhi{C}, f, θ, training_mapping::Tuple{Vector, Vector}) where {C}
    input_set, _ = training_mapping
    data_set_size = size(input_set)[1] * size(input_set[1])[2]
    loss = reduce(vcat,
        [inner_physics_loss(phi, f, θ, in_) for in_ in input_set])
    sum(abs2, loss) / data_set_size
end

function data_loss(phi::PINOPhi{C}, θ, training_mapping::Tuple{Vector, Vector}) where {C}
    input_set, output_set = training_mapping
    data_set_size = size(input_set)[1] * size(input_set[1])[2]
    inner_data_loss(phi::PINOPhi{C}, θ, in_, out_) where {C}  = phi(in_, θ) - out_
    loss = reduce(vcat,
        [inner_data_loss(phi, θ, in_, out_) for (in_, out_) in zip(input_set, output_set)])
    sum(abs2,loss) / data_set_size
end

function generate_loss(phi::PINOPhi{C}, f, training_mapping::Tuple{Vector, Vector}) where {C}
    function loss(θ, _)
        data_loss(phi, θ, training_mapping) + physics_loss(phi, f, θ, training_mapping)
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
    # u0 = prob.u0 ? TODO
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    # param_estim = alg.param_estim

    chain = alg.chain
    opt = alg.opt
    init_params = alg.init_params

    # mapping between functional space of some vararible 'a' of equation (for example initial
    # condition {u(t0 x)} or parameter) join  and solution of equation u(t)
    training_mapping = alg.training_mapping

    !(chain isa Lux.AbstractExplicitLayer) &&
        error("Only Lux.AbstractExplicitLayer neural networks are supported")

    phi, init_params = generate_pino_phi_θ(chain, init_params)

    init_params = ComponentArrays.ComponentArray(init_params)

    isinplace(prob) &&
        throw(error("The PINOODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))

    try
        phi(first(training_mapping[1]), init_params)
    catch err
        if isa(err, DimensionMismatch)
            throw(DimensionMismatch("Dimensions of the initial u0 and chain should match"))
        else
            throw(err)
        end
    end

    # batch = if alg.batch === nothing
    #     true
    # else
    #     alg.batch
    # end

    inner_f = generate_loss(phi, f, training_mapping)

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
