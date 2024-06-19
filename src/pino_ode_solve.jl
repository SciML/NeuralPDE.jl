struct ParametricFunction{}
    function_ ::Union{Nothing, Function}
    bounds::Any
end

"""
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
                 `Flux.Chain` will be converted to `Lux` using `adapt(FromFluxAdaptor(false, false), chain)`
* `opt`: The optimizer to train the neural network.
* `bounds`: A dictionary containing the bounds for the parameters of the neural network
in which will be train the prediction of parametric ODE.

## Keyword Arguments

* `init_params`: The initial parameter of the neural network. By default, this is `nothing`,
                             which thus uses the random initialization provided by the neural network library.
* `strategy`: The strategy for training the neural network.
* `additional_loss`: additional function to the main one. For example, add training on data.
* `kwargs`: Extra keyword arguments are splatted to the Optimization.jl `solve` call.

## References

* Sifan Wang "Learning the solution operator of parametric partial differential equations with physics-informed DeepOnets"
* Zongyi Li "Physics-Informed Neural Operator for Learning Partial Differential Equations"
"""
struct PINOODE{C, O, I, S <: Union{Nothing, AbstractTrainingStrategy},
    AL <: Union{Nothing, Function}, K} <:
       SciMLBase.AbstractODEAlgorithm
    chain::C
    opt::O
    parametric_function::ParametricFunction
    init_params::I
    strategy::S
    additional_loss::AL
    kwargs::K
end

function PINOODE(chain,
        opt,
        parametric_function;
        init_params = nothing,
        strategy = nothing,
        additional_loss = nothing,
        kwargs...)
    !(chain isa Lux.AbstractExplicitLayer) && (chain = Lux.transform(chain))
    PINOODE(chain, opt, parametric_function, init_params, strategy, additional_loss, kwargs)
end

mutable struct PINOPhi{C, S}
    chain::C
    st::S
    function PINOPhi(chain::Lux.AbstractExplicitLayer, st)
        new{typeof(chain), typeof(st)}(chain, st)
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

function (f::PINOPhi{C, T})(x::NamedTuple, θ) where {C <: NeuralOperator, T}
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), x), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    y
end

function dfdx(phi::PINOPhi{C, T}, x::Tuple, θ, prob::ODEProblem) where {C <: DeepONet, T}
    pfs, p, t = x
    # branch_left, branch_right = pfs, pfs
    trunk_left, trunk_right = t .+ sqrt(eps(eltype(t))), t
    x_left = (branch = pfs, trunk = trunk_left)
    x_right = (branch = pfs, trunk = trunk_right)
    (phi(x_left, θ) .- phi(x_right, θ)) / sqrt(eps(eltype(t)))
end

# function physics_loss(
#         phi::PINOPhi{C, T}, prob::ODEProblem, x, θ) where {C <: DeepONet, T}
#     p, t = x
#     f = prob.f
#     du = vec(dfdx(phi, x, θ, prob))
#     f_ = f.(0, p, t)
#     tuple = (branch = f_, trunk = t)
#     out = phi(tuple, θ)
#     f_ = vec(f.(out, p, t))
#     norm = prod(size(out))
#     sum(abs2, du .- f_) / norm
# end
# function initial_condition_loss(
#         phi::PINOPhi{C, T}, prob::ODEProblem, x, θ) where {C <: DeepONet, T}
#     p, t = x
#     f = prob.f
#     t0 = t[:, :, [1]]
#     f_0 = f.(0, p, t0)
#     tuple = (branch = f_0, trunk = t0)
#     out = phi(tuple, θ)
#     u = vec(out)
#     u0_ = fill(prob.u0, size(out))
#     u0 = vec(u0_)
#     norm = prod(size(u0))
#     sum(abs2, u .- u0) / norm
# end

function physics_loss(
        phi::PINOPhi{C, T}, prob::ODEProblem, x, θ) where {C <: DeepONet, T}
    pfs, p, t = x
    f = prob.f
    du = vec(dfdx(phi, x, θ, prob))
    tuple = (branch = pfs, trunk = t)
    out = phi(tuple, θ)
    # if size(p)[1] == 1
        fs = f.(out, p, t)
        f_ = vec(fs)
    # else
    #     f_ = reduce(vcat,[reduce(vcat, [f(out[i], p[i], t[j]) for i in axes(p, 2)]) for j in axes(t, 3)])
    # end
    norm = prod(size(out))
    sum(abs2, du .- f_) / norm
end

function initial_condition_loss(phi::PINOPhi{C, T}, prob::ODEProblem, x, θ) where {C <: DeepONet, T}
    pfs, p, t = x
    t0 = t[:, :, [1]]
    pfs0 = pfs[:, :, [1]]
    tuple = (branch = pfs0, trunk = t0)
    out = phi(tuple, θ)
    u = vec(out)
    u0 = vec(fill(prob.u0, size(out)))
    norm = prod(size(u0))
    sum(abs2, u .- u0) / norm
end

# function get_trainset(strategy::GridTraining, bounds, tspan)
#     db, dt = strategy.dx
#     v  = values(bounds)[1]
#     #TODO for all v
#     p_ = v[1]:db:v[2]
#     p = reshape(p_, 1, size(p_)[1], 1)
#     t_ = collect(tspan[1]:dt:tspan[2])
#     t = reshape(t_, 1, 1, size(t_)[1])
#     (p, t)
# end

function get_trainset(
        strategy::GridTraining, parametric_function::ParametricFunction, tspan)
    @unpack function_, bounds = parametric_function
    dt = strategy.dx
    #TODO
    size_of_p = 50
    if bounds isa Tuple
        p_ = range(start = bounds[1], length = size_of_p, stop = bounds[2])
        p = collect(reshape(p_, 1, size(p_)[1], 1))
    else
        p_ = [range(start = b[1], length = size_of_p, stop = b[2]) for b in bounds]
        p = vcat([collect(reshape(p_i, 1, size(p_i)[1], 1)) for p_i in p_]...)
    end

    t_ = collect(tspan[1]:dt:tspan[2])
    t = reshape(t_, 1, 1, size(t_)[1])
    pfs = function_.(p,t)
    (pfs, p, t)
end

function generate_loss(
        strategy::GridTraining, prob::ODEProblem, phi, parametric_function::ParametricFunction, tspan)
    x = get_trainset(strategy, parametric_function, tspan)
    function loss(θ, _)
        initial_condition_loss(phi, prob, x, θ) + physics_loss(phi, prob, x, θ)
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
    @unpack chain, opt, parametric_function, init_params, strategy, additional_loss = alg

    if !isa(chain, DeepONet)
        error("Only DeepONet neural networks are supported")
    end

    !(chain isa Lux.AbstractExplicitLayer) &&
        error("Only Lux.AbstractExplicitLayer neural networks are supported")

    phi, init_params = generate_pino_phi_θ(chain, init_params)

    isinplace(prob) &&
        throw(error("The PINOODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))

    try
        x = (branch = rand(1, 10, 10), trunk = rand(1, 1, 10))
        phi(x, init_params)
    catch err
        if isa(err, DimensionMismatch)
            throw(DimensionMismatch("Dimensions of input data and chain should match"))
        else
            throw(err)
        end
    end

    if strategy === nothing
        dt = (tspan[2] - tspan[1]) / 50
        strategy = GridTraining(dt)
    elseif !isa(strategy, GridTraining)
        throw(ArgumentError("Only GridTraining strategy is supported"))
    end

    inner_f = generate_loss(strategy, prob, phi, parametric_function, tspan)

    function total_loss(θ, _)
        L2_loss = inner_f(θ, nothing)
        if !(additional_loss isa Nothing)
            L2_loss = L2_loss + additional_loss(phi, θ)
        end
        L2_loss
    end

    # TODO delete
    # total_loss(θ, 0)
    # Zygote.gradient(θ -> total_loss(θ, 0), θ)

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

    pfs, p, t = get_trainset(strategy, parametric_function, tspan)
    tuple = (branch = pfs, trunk = t)
    u = phi(tuple, res.u)

    sol = SciMLBase.build_solution(prob, alg, tuple, u;
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
