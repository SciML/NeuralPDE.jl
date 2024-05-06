"""
 PINOODE

PINOODE(chain,
    OptimizationOptimisers.Adam(0.1),
    bounds;
    init_params = nothing,
    strategy = nothing
    kwargs...)

Algorithm for solving paramentric ordinary differential equations using a physics-informed
neural operator, which is used as a solver for a parametrized `ODEProblem`.

## Positional Arguments
* `chain`: A neural network architecture, defined as a `Lux.AbstractExplicitLayer` or `Flux.Chain`.
          `Flux.Chain` will be converted to `Lux` using `Lux.transform`.
* `opt`: The optimizer to train the neural network.
* `bounds`: A dictionary containing the bounds for the parameters of the neural network
in which will be train the prediction of parametric ODE.

## Keyword Arguments
* `init_params`: The initial parameter of the neural network. By default, this is `nothing`
  which thus uses the random initialization provided by the neural network library.
* `strategy`: The strategy for training the neural network.
* `additional_loss`: additional function to the main one. For example, add training on data.
* `kwargs`: Extra keyword arguments are splatted to the Optimization.jl `solve` call.

## References
* Sifan Wang "Learning the solution operator of parametric partial differential equations with physics-informed DeepOnets"
* Zongyi Li "Physics-Informed Neural Operator for Learning Partial Differential Equations"
"""
struct PINOODE{C, O, B, I, S <: Union{Nothing, AbstractTrainingStrategy}, AL <: Union{Nothing, Function},K} <:
       SciMLBase.AbstractODEAlgorithm
    chain::C
    opt::O
    bounds::B
    init_params::I
    strategy::S
    additional_loss::AL #TODO
    kwargs::K
end

function PINOODE(chain,
        opt,
        bounds;
        init_params = nothing,
        strategy = nothing,
        additional_loss = nothing,
        kwargs...)
    !(chain isa Lux.AbstractExplicitLayer) && (chain = Lux.transform(chain))
    PINOODE(chain, opt, bounds, init_params, strategy, additional_loss, kwargs)
end

#TODO change to GridStrategy
struct SomeStrategy{} <: AbstractTrainingStrategy
    branch_size::Int
    trunk_size::Int
end

function SomeStrategy(;branch_size = 10, trunk_size=10)
    SomeStrategy(branch_size, trunk_size)
end

mutable struct PINOPhi{C, T, U, S}
    chain::C
    t0::T
    u0::U#TODO remove u0, t0?
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

function (f::PINOPhi{C, T, U})(x::NamedTuple, θ) where {C, T, U}
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), x), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    y
end

# function dfdx(phi::PINOPhi{C, T, U}, x::NamedTuple, θ, branch_left) where {C, T, U}
#     x_trunk = x.trunk
#     x_left = (branch = branch_left, trunk = x_trunk .+ sqrt(eps(eltype(x_trunk))))
#     x_right = (branch = x.branch, trunk = x_trunk)
#     (phi(x_left, θ) .- phi(x_right, θ)) / sqrt(eps(eltype(x_trunk)))
# end

#TODO C <: DeepONet
function dfdx(phi::PINOPhi{C, T, U}, x::Tuple, θ, prob::ODEProblem) where {C, T, U}
    p, t = x
    f = prob.f
    branch_left, branch_right = f.(0, p, t .+ sqrt(eps(eltype(p)))),  f.(0, p, t)
    trunk_left, trunk_right = t .+ sqrt(eps(eltype(t))), t
    x_left = (branch = branch_left, trunk = trunk_left)
    x_right = (branch = branch_right, trunk = trunk_right)
    (phi(x_left, θ) .- phi(x_right, θ)) / sqrt(eps(eltype(t)))
end

function physics_loss(phi::PINOPhi{C, T, U}, prob::ODEProblem, x, θ) where {C, T, U}
    p, t = x
    f = prob.f
    #TODO If du = f(u,p,t), where f = g(p,t)*u so it will wrong, f(0, p, t) = g(p,t)*0 = 0
    #work correct only with function like du = f(p,t) + g(u)
    du = vec(dfdx(phi, x, θ, prob))

    tuple = (branch = f.(0, p, t), trunk = t)
    out = phi(tuple, θ)
    f_ = vec(f.(out, p, t))
    norm = prod(size(out))
    sum(abs2, du .- f_) / norm
end

function operator_loss(phi::PINOPhi{C, T, U}, prob::ODEProblem, x, θ) where {C, T, U}
    p, t = x
    f = prob.f
    t0 = t[:, :, [1]]
    f_0 = f.(0, p, t0)
    tuple = (branch = f_0, trunk = t0)
    out = phi(tuple, θ)
    u = vec(out)
    u0_ = fill(prob.u0, size(out))
    u0 = vec(u0_)
    norm = prod(size(u0_))
    sum(abs2, u .- u0) / norm
end

function get_trainset(branch_size, trunk_size, bounds, tspan, prob)
    p_ = range(bounds.p[1], stop = bounds.p[2], length = branch_size)
    p = reshape(p_, 1, branch_size,1)
    t_ = collect(range(tspan[1], stop = tspan[2], length = trunk_size))
    t = reshape(t_, 1, 1, trunk_size)
    (p,t)
end

function generate_loss(strategy, prob::ODEProblem, phi, bounds, tspan)
    branch_size, trunk_size = strategy.branch_size, strategy.trunk_size
    x = get_trainset(branch_size, trunk_size, bounds, tspan, prob)
    function loss(θ, _)
        physics_loss(phi, prob, x, θ) + operator_loss(phi, prob, x, θ)
    end
end

function SciMLBase.__solve(prob::SciMLBase.AbstractODEProblem,
        alg::PINOODE,
        args...;
        abstol = 1.0f-8,
        reltol = 1.0f-3,
        verbose = false,
        saveat = nothing,
        maxiters = nothing)
    @unpack tspan, u0, f = prob
    @unpack chain, opt, bounds, init_params, strategy = alg

    !(chain isa Lux.AbstractExplicitLayer) &&
        error("Only Lux.AbstractExplicitLayer neural networks are supported")

    #TODO support for u0
    if !any(in(keys(bounds)), (:p,))
        error("bounds should contain p only")
    end

    phi, init_params = generate_pino_phi_θ(chain, 0, u0, init_params)

    isinplace(prob) &&
        throw(error("The PINOODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))

    try
        x = (branch = rand(1, 10,10), trunk = rand(1, 1, 10))
        phi(x, init_params)
    catch err
        if isa(err, DimensionMismatch)
            throw(DimensionMismatch("Dimensions of input data and chain should match"))
        else
            throw(err)
        end
    end

    inner_f = generate_loss(strategy, prob, phi, bounds, tspan)

    #TODO impl add loss
    function total_loss(θ, _)
        inner_f(θ, nothing)

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

    branch_size, trunk_size = strategy.branch_size, strategy.trunk_size
    p, t = get_trainset(branch_size, trunk_size, bounds, tspan, prob)
    x = (branch = f.(0, p,t), trunk = t)
    u = phi(x, res.u)

    sol = SciMLBase.build_solution(prob, alg, x, u;
        k = res, dense = true,
        calculate_error = false,
        retcode = ReturnCode.Success,
        original = res,
        resid = res.objective)
    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end
