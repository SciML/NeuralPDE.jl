"""
   PINOODE(chain,
    OptimizationOptimisers.Adam(0.1),
    pino_phase;
    init_params,
    kwargs...)

The method is that combine training data and physics constraints
to learn the solution operator of a given family of parametric Ordinary Differential Equations (ODE).

## Positional Arguments
* `chain`: A neural network architecture, defined as a `Lux.AbstractExplicitLayer` or `Flux.Chain`.
          `Flux.Chain` will be converted to `Lux` using `Lux.transform`.
* `opt`: The optimizer to train the neural network.
* `pino_phase`: The phase of the PINN algorithm, either `OperatorLearning` or `EquationSolving`.

## Keyword Arguments
* `init_params`: The initial parameter of the neural network. By default, this is `nothing`
  which thus uses the random initialization provided by the neural network library.
* isu0: If true, the input data set contains initial conditions 'u0'.
* `kwargs`: Extra keyword arguments are splatted to the Optimization.jl `solve` call.

## References
* Sifan Wang "Learning the solution operator of parametric partial differential equations with physics-informed DeepOnets"
* Zongyi Li "Physics-Informed Neural Operator for Learning Partial Differential Equations"
"""
struct PINOODE{C, O, B, I, S, K} <: SciMLBase.AbstractODEAlgorithm
    chain::C
    opt::O
    bounds::B
    init_params::I
    isu0::Bool
    strategy::S
    kwargs::K
end

function PINOODE(chain,
        opt,
        bounds;
        init_params = nothing,
        isu0 = false, #TODOD remove
        strategy = nothing,
        kwargs...)
    !(chain isa Lux.AbstractExplicitLayer) && (chain = Lux.transform(chain))
    PINOODE(chain, opt, bounds, init_params, isu0, strategy, kwargs)
end

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

#TODO update
# function (f::PINOPhi{C, T, U})(t::AbstractArray,
#         θ) where {C <: Lux.AbstractExplicitLayer, T, U}
#     y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), t), θ, f.st)
#     ChainRulesCore.@ignore_derivatives f.st = st
#     ts = adapt(parameterless_type(ComponentArrays.getdata(θ)), t[1:size(y)[1], :, :])
#     u_0 = adapt(parameterless_type(ComponentArrays.getdata(θ)), f.u0)
#     u_0 .+ (ts .- f.t0) .* y
# end

#TODO C <: DeepONet
function (f::PINOPhi{C, T, U})(x::NamedTuple, θ) where {C, T, U}
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), x), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    a, t = x.branch, x.trunk
    ts = adapt(parameterless_type(ComponentArrays.getdata(θ)), t)
    u0_ = adapt(parameterless_type(ComponentArrays.getdata(θ)), f.u0)
    u0_ .+ (ts .- f.t0) .* y
end

# function dfdx(phi::PINOPhi, t::AbstractArray, θ)
#     ε = [sqrt(eps(eltype(t))), zeros(eltype(t), size(t)[1] - 1)...]
#     (phi(t .+ ε, θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
# end

#TODO C <: DeepONet
function dfdx(phi::PINOPhi{C, T, U}, x::NamedTuple, θ) where {C, T, U}
    t = x.trunk
    ε = [sqrt(eps(eltype(t)))]
    phi_trunk(x, θ) = phi.chain.trunk(x, θ.trunk, phi.st.trunk)[1]
    du_trunk_ = (phi_trunk(t .+ ε, θ) .- phi_trunk(t, θ)) ./ sqrt(eps(eltype(t)))
    u_branch = phi.chain.branch(x.branch, θ.branch, phi.st.branch)[1]
    u_branch' .* du_trunk_
end

# function l₂loss(𝐲̂, 𝐲)
#     feature_dims = 2:(ndims(𝐲) - 1)
#     loss = sum(.√(sum(abs2, 𝐲̂ - 𝐲, dims = feature_dims)))
#     y_norm = sum(.√(sum(abs2, 𝐲, dims = feature_dims)))
#     return loss / y_norm
# end

function physics_loss(phi::PINOPhi{C, T, U}, prob::ODEProblem, x, θ) where {C, T, U}
    f = prob.f
    ps , ts  = x.branch, x.trunk
    norm = size(x.branch)[2] * size(x.trunk)[2]
    sum(abs2, dfdx(phi, x, θ) - f.(phi(x, θ), ps, ts)) / norm
end

function get_trainset(bounds, tspan, strategy)
    #TODO dt -> instances_size
    instances_size = 100
    p = range(bounds.p[1], stop = bounds.p[2], length = instances_size)
    t = range(tspan[1], stop = tspan[2], length = instances_size)
    x = (branch = collect(p)', trunk = collect(t)')
    x
end

#TODO GridTraining
function generate_loss(strategy, prob::ODEProblem, phi, bounds, tspan)
    x = get_trainset(bounds, tspan, strategy)
    function loss(θ, _)
        physics_loss(phi, prob, x, θ)
    end
end

function SciMLBase.__solve(prob::SciMLBase.AbstractODEProblem,
        alg::PINOODE,
        args...;
        dt = nothing,
        abstol = 1.0f-6,
        reltol = 1.0f-3,
        verbose = false,
        saveat = nothing,
        maxiters = nothing)
    @unpack tspan, u0, p, f = prob
    t0, t_end = tspan[1], tspan[2]
    @unpack chain, opt, bounds, init_params, isu0 = alg

    !(chain isa Lux.AbstractExplicitLayer) &&
        error("Only Lux.AbstractExplicitLayer neural networks are supported")

    if !any(in(keys(bounds)), (:u0, :p))
        error("bounds should contain u0 or p only")
    end

    phi, init_params = generate_pino_phi_θ(chain, t0, u0, init_params)
    init_params = ComponentArrays.ComponentArray(init_params)

    isinplace(prob) &&
        throw(error("The PINOODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))

    try
        x = (branch = rand(length(bounds), 10), trunk = rand(1, 10))
        phi(x, init_params)
    catch err
        if isa(err, DimensionMismatch)
            throw(DimensionMismatch("Dimensions of input data and chain should match"))
        else
            throw(err)
        end
    end

    strategy = nothing

    inner_f = generate_loss(strategy, prob, phi, bounds, tspan)

    function total_loss(θ, _)
        inner_f(θ, nothing)
        #TODO add loss
        # L2_loss = inner_f(θ, nothing)
        # if !(additional_loss isa Nothing)
        #     L2_loss = L2_loss + additional_loss(phi, θ)
        # end
    end

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
    res, phi

    #TODO build_solution
    # if saveat isa Number
    #     ts = tspan[1]:saveat:tspan[2]
    # end

    # if u0 isa Number
    #     u = [first(phi(t, res.u)) for t in ts]
    # end
    # sol = SciMLBase.build_solution(prob, alg, ts, u;
    # sol
end
