"""
	PINOODE(chain,
	    opt,
	    bounds;
	    init_params = nothing,
	    strategy = nothing
	    kwargs...)

Algorithm for solving paramentric ordinary differential equations using a physics-informed
neural operator, which is used as a solver for a parametrized `ODEProblem`.

## Positional Arguments

* `chain`: A neural network architecture, defined as a `Lux.AbstractLuxLayer` or `Flux.Chain`.
                 `Flux.Chain` will be converted to `Lux` using `adapt(FromFluxAdaptor(false, false), chain)`
* `opt`: The optimizer to train the neural network.
* `bounds`: A dictionary containing the bounds for the parameters of the parametric ODE.
* `number_of_parameters`: The number of points of train set in parameters boundaries.

## Keyword Arguments

* `init_params`: The initial parameters of the neural network. By default, this is `nothing`,
                             which thus uses the random initialization provided by the neural network library.
* `strategy`: The strategy for training the neural network.
* `additional_loss`: additional loss function added to the default one. For example, add training on data.
* `kwargs`: Extra keyword arguments are splatted to the Optimization.jl `solve` call.

## References

* Sifan Wang "Learning the solution operator of parametric partial differential equations with physics-informed DeepOnets"
* Zongyi Li "Physics-Informed Neural Operator for Learning Partial Differential Equations"
"""
struct PINOODE{C, O, B, I, S <: Union{Nothing, AbstractTrainingStrategy},
    AL <: Union{Nothing, Function}, K} <:
       SciMLBase.AbstractODEAlgorithm
    chain::C
    opt::O
    bounds::B
    number_of_parameters::Int
    init_params::I
    strategy::S
    additional_loss::AL
    kwargs::K
end

function PINOODE(chain,
        opt,
        bounds,
        number_of_parameters;
        init_params = nothing,
        strategy = nothing,
        additional_loss = nothing,
        kwargs...)
    !(chain isa Lux.AbstractLuxLayer) && (chain = Lux.transform(chain))
    PINOODE(chain, opt, bounds, number_of_parameters,
        init_params, strategy, additional_loss, kwargs)
end

struct PINOPhi{C, S}
    chain::C
    st::S
    function PINOPhi(chain::Lux.AbstractLuxLayer, st)
        new{typeof(chain), typeof(st)}(chain, st)
    end
end

function generate_pino_phi_θ(chain::Lux.AbstractLuxLayer, init_params)
    θ, st = Lux.setup(Random.default_rng(), chain)
    init_params = isnothing(init_params) ? θ : init_params
    init_params = ComponentArrays.ComponentArray(init_params)
    PINOPhi(chain, st), init_params
end

function (f::PINOPhi{C, T})(
        x, θ) where {C <: Lux.AbstractLuxLayer, T}
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), x), θ, f.st)
    y
end

function dfdx(phi::PINOPhi{C, T}, x::Tuple, θ) where {C <: DeepONet, T}
    p, t = x
    branch_left, branch_right = p, p
    trunk_left, trunk_right = t .+ sqrt(eps(eltype(t))), t
    x_left = (branch_left, trunk_left)
    x_right = (branch_right, trunk_right)
    (phi(x_left, θ) .- phi(x_right, θ)) ./ sqrt(eps(eltype(t)))
end

function physics_loss(
        phi::PINOPhi{C, T}, prob::ODEProblem, x::Tuple, θ) where {C <: DeepONet, T}
    p, t = x
    f = prob.f
    out = phi(x, θ)
    if size(p, 1) == 1
        f_vec = vec(f.(out, p, vec(t)))
    else
        f_vec = reduce(
            vcat, [[f(out[i], p[:, i], t[j]) for j in axes(t, 2)] for i in axes(p, 2)])
    end
    du = vec(dfdx(phi, x, θ))
    norm = prod(size(du))
    sum(abs2, du .- f_vec) / norm
end

function initial_condition_loss(
        phi::PINOPhi{C, T}, prob::ODEProblem, x, θ) where {
        C <: DeepONet, T}
    p, t = x
    t0 = reshape([prob.tspan[1]], (1, 1, 1))
    x0 = (p, t0)
    out = phi(x0, θ)
    u = vec(out)
    u0 = vec(fill(prob.u0, size(out)))
    norm = prod(size(u0))
    sum(abs2, u .- u0) / norm
end

function get_trainset(
        strategy::GridTraining, chain::DeepONet, bounds, number_of_parameters, tspan)
    dt = strategy.dx
    p_ = [range(start = b[1], length = number_of_parameters, stop = b[2]) for b in bounds]
    p = vcat([collect(reshape(p_i, 1, size(p_i, 1))) for p_i in p_]...)
    t_ = collect(tspan[1]:dt:tspan[2])
    t = reshape(t_, 1, size(t_, 1), 1)
    (p, t)
end

function get_trainset(
        strategy::StochasticTraining, chain::DeepONet, bounds, number_of_parameters, tspan)
    p = reduce(vcat,
        [(bound[2] .- bound[1]) .* rand(1, number_of_parameters) .+ bound[1]
         for bound in bounds])
    t = (tspan[2] .- tspan[1]) .* rand(1, strategy.points, 1) .+ tspan[1]
    (p, t)
end

function generate_loss(
        strategy::GridTraining, prob::ODEProblem, phi, bounds, number_of_parameters, tspan)
    x = get_trainset(strategy, phi.chain, bounds, number_of_parameters, tspan)
    function loss(θ, _)
        initial_condition_loss(phi, prob, x, θ) + physics_loss(phi, prob, x, θ)
    end
end

function generate_loss(
        strategy::StochasticTraining, prob::ODEProblem, phi, bounds, number_of_parameters, tspan)
    function loss(θ, _)
        x = get_trainset(strategy, phi.chain, bounds, number_of_parameters, tspan)
        initial_condition_loss(phi, prob, x, θ) + physics_loss(phi, prob, x, θ)
    end
end

struct PINOODEInterpolation{T <: PINOPhi, T2}
    phi::T
    θ::T2
end

(f::PINOODEInterpolation)(x) = f.phi(x, f.θ)

SciMLBase.interp_summary(::PINOODEInterpolation) = "Trained neural network interpolation"
SciMLBase.allowscomplex(::PINOODE) = true

function SciMLBase.__solve(prob::SciMLBase.AbstractODEProblem,
        alg::PINOODE,
        args...;
        abstol = 1.0f-8,
        reltol = 1.0f-3,
        verbose = false,
        saveat = nothing,
        maxiters = nothing)
    @unpack tspan, u0, f = prob
    @unpack chain, opt, bounds, number_of_parameters, init_params, strategy, additional_loss = alg

    if !(chain isa Lux.AbstractLuxLayer)
        error("Only Lux.AbstractLuxLayer neural networks are supported")

        if !(chain isa DeepONet) #|| chain isa FourierNeuralOperator)
            error("Only DeepONet and FourierNeuralOperator neural networks are supported with PINO ODE")
        end
    end

    phi, init_params = generate_pino_phi_θ(chain, init_params)

    isinplace(prob) &&
        throw(error("The PINOODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))

    try
        if chain isa DeepONet
            in_dim = chain.branch.layers.layer_1.in_dims
            u = rand(Float32, in_dim, number_of_parameters)
            v = rand(Float32, 1, 10, 1)
            x = (u, v)
            phi(x, init_params)
        end
    catch err
        if isa(err, DimensionMismatch)
            throw(DimensionMismatch("Dimensions of input data and chain should match"))
        else
            throw(err)
        end
    end

    if strategy === nothing
        strategy = StochasticTraining(100)
    elseif !(strategy isa GridTraining || strategy isa StochasticTraining)
        throw(ArgumentError("Only GridTraining and StochasticTraining strategy is supported"))
    end

    inner_f = generate_loss(strategy, prob, phi, bounds, number_of_parameters, tspan)

    function total_loss(θ, _)
        L2_loss = inner_f(θ, nothing)
        if !(additional_loss isa Nothing)
            L2_loss = L2_loss + additional_loss(phi, θ)
        end
        L2_loss
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

    x = get_trainset(strategy, phi.chain, bounds, number_of_parameters, tspan)
    u = phi(x, res.u)

    sol = SciMLBase.build_solution(prob, alg, x, u;
        k = res, dense = true,
        interp = PINOODEInterpolation(phi, res.u),
        calculate_error = false,
        retcode = ReturnCode.Success,
        original = res,
        resid = res.objective)
    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end
