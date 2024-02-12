struct PINOODE{C, O, T, P, B, K} <: DiffEqBase.AbstractODEAlgorithm
    chain::C
    opt::O
    training_mapping::T
    init_params::P
    batch::B
    kwargs::K
end

function PINOODE(chain,
        opt,
        training_mapping,
        init_params = nothing;
        batch = nothing,
        kwargs...)
    !(chain isa Lux.AbstractExplicitLayer) && (chain = Lux.transform(chain))
    PINOODE(chain, opt, training_mapping, init_params, batch, kwargs)
end

"""
    PINOPhi(chain::Lux.AbstractExplicitLayer, t, u0, st)
"""
mutable struct PINOPhi{C, T, U, S}
    chain::C
    t0::T
    u0::U
    st::S
    function PINOPhi(chain::Lux.AbstractExplicitLayer, t::Number, u0, st)
        new{typeof(chain), typeof(t), typeof(u0), typeof(st)}(chain, t, u0, st)
    end
end

function generate_pino_phi_θ(chain::Lux.AbstractExplicitLayer, t, u0, init_params)
    θ, st = Lux.setup(Random.default_rng(), chain)
    if init_params === nothing
        init_params = ComponentArrays.ComponentArray(θ)
    else
        init_params = ComponentArrays.ComponentArray(init_params)
    end
    PINOPhi(chain, t, u0, st), init_params
end

function (f::PINOPhi{C, T, U})(t::AbstractMatrix,
        θ) where {C <: Lux.AbstractExplicitLayer, T, U}
    # Batch via data as row vectors
    # y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ.depvar)), t'),
        # θ.depvar,
        # f.st)
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), t),
        θ,
        f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    # f.u0 .+ (t' .- f.t0) .* y
    y
end

function inner_data_loss(phi, ts, θ, a, ground_u, i)
    u_size = size(ground_u[i])
    a_arr = fill(a[i], u_size)
    input_data = reduce(vcat, [ts, a_arr])
    y = phi(input_data, θ)
    y - ground_u[i]
end
function data_loss(phi,tspan, θ, training_mapping)
    a, ground_u = training_mapping
    t0, t_end = tspan
    size_1 = size(ground_u[1])
    size_2 = size(ground_u)[1]
    ts = Float32.(reshape(collect(range(t0, stop = t_end, length = size_1[2])), size_1))
    loss = [inner_data_loss(phi, ts, θ, a, ground_u,i)
            for i in 1:size_2]
    # reduce(vcat, loss)
    sum(abs2, reduce(vcat, loss)) / (size_1[2] * size_2)
end

function inner_loss(phi, tspan, θ, training_mapping)
    loss = data_loss(phi, tspan, θ, training_mapping ) #+ physics_loss()
    loss
end

function generate_loss(phi, tspan, training_mapping) #training_mapping::Tuple
    function loss(θ, _)
        # inner_loss(phi, f, tspan, θ, p,training_mapping )
        sum(abs2, inner_loss(phi, tspan, θ, training_mapping))
    end
    return loss
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem,
        alg::PINOODE,
        args...;
        dt = nothing,
        abstol = 1.0f-6,
        reltol = 1.0f-3,
        verbose = false,
        saveat = nothing,
        maxiters = nothing)
    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t0 = tspan[1]
    # param_estim = alg.param_estim

    #hidden layer
    chain = alg.chain
    opt = alg.opt

    #train points generation
    init_params = alg.init_params

    # mapping between functional space of some vararible 'a' of equation (for example initial
    # codition {u(t0 x)} or parameter) and solution of equation u(t)
    training_mapping = alg.training_mapping

    !(chain isa Lux.AbstractExplicitLayer) &&
        error("Only Lux.AbstractExplicitLayer neural networks are supported")

    phi, init_params = generate_pino_phi_θ(chain, t0, u0, init_params)

    # init_params = ComponentArrays.ComponentArray(init_params)
    # init_params = ComponentArrays.ComponentArray(;depvar = ComponentArrays.ComponentArray(init_params))
    @show phi(rand(2,1), init_params)
    @show phi(rand(2, 10), init_params)

    isinplace(prob) &&
        throw(error("The PINOODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))

    try
        # TODO
        phi(rand(2, 1), init_params)
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

    inner_f = generate_loss(phi, tspan, training_mapping)

    # Creates OptimizationFunction Object from total_loss
    total_loss(θ, _) = inner_f(θ, phi)


    # Optimization Algo for Training Strategies
    opt_algo = Optimization.AutoZygote()

    # Creates OptimizationFunction Object from total_loss
    optf = OptimizationFunction(total_loss, opt_algo)

    # iteration = 0
    # callback = function (p, l)
    #     iteration += 1
    #     verbose && println("Current loss is: $l, Iteration: $iteration")
    #     l < abstol
    # end

    callback = function (p, l)
        println("Current loss is: $l")
        return false
    end

    # init_params = ComponentArrays.ComponentArray(θ)

    # Zygote.gradient(θ -> total_loss(θ, 1), init_params)

    optprob = OptimizationProblem(optf, init_params)
    res = solve(optprob, opt; callback = callback, maxiters = maxiters) # alg.kwargs...)

    # PINOsolution(fullsolution)
    # res
    (total_loss,optprob,opt,callback, maxiters)
end
