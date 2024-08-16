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

* `chain`: A neural network architecture, defined as a `Lux.AbstractExplicitLayer` or `Flux.Chain`.
                 `Flux.Chain` will be converted to `Lux` using `adapt(FromFluxAdaptor(false, false), chain)`
* `opt`: The optimizer to train the neural network.
* `bounds`: A dictionary containing the bounds for the parameters of the neural network
in which will be train the prediction of parametric ODE.
* `number_of_parameters`: The number of points of train set in parameters boundaries.

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
    !(chain isa Lux.AbstractExplicitLayer) && (chain = Lux.transform(chain))
    PINOODE(chain, opt, bounds, number_of_parameters,
        init_params, strategy, additional_loss, kwargs)
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
    init_params = isnothing(init_params) ? θ : init_params
    # init_params = chain.name == "FourierNeuralOperator" ? θ : ComponentArrays.ComponentArray(init_params)
    init_params = ComponentArrays.ComponentArray(init_params)
    PINOPhi(chain, st), init_params
end

function (f::PINOPhi{C, T})(
        x, θ) where {C <: Lux.AbstractExplicitLayer, T}
    y, st = f.chain(adapt(parameterless_type(ComponentArrays.getdata(θ)), x), θ, f.st)
    ChainRulesCore.@ignore_derivatives f.st = st
    y
end

function dfdx(phi::PINOPhi{C, T}, x::Tuple, θ) where {C <: CompactLuxLayer{:DeepONet,}, T}
    p, t = x
    branch_left, branch_right = p, p
    trunk_left, trunk_right = t .+ sqrt(eps(eltype(t))), t
    x_left = (branch_left, trunk_left)
    x_right = (branch_right, trunk_right)
    (phi(x_left, θ) .- phi(x_right, θ)) ./ sqrt(eps(eltype(t)))
end

#FourierNeuralOperator and Chain
function dfdx(phi::PINOPhi{C, T}, x::Array, θ) where {C, T}
    ε = [zeros(eltype(x), size(x)[1] - 1)..., sqrt(eps(eltype(x)))]
    (phi(t .+ ε, θ) - phi(t, θ)) ./ sqrt(eps(eltype(t)))
end

function physics_loss(
        phi::PINOPhi{C, T}, prob::ODEProblem, x::Tuple, θ) where {
        C <: CompactLuxLayer{:DeepONet,}, T}
    p, t = x
    f = prob.f
    out = phi(x, θ)
    # if size(p,1) == 1
        #TODO
        if size(out)[1] == 1
            out = dropdims(out, dims = 1)
        end
        # out = dropdims(out, dims = 1)
        fs = f.(out, p, vec(t))
        f_vec = vec(fs)
    # else
    #     # f_vec = reduce(
    #     #     vcat, [[f(out[i], p[:, i], t[j]) for j in axes(t, 2)] for i in axes(p, 2)])
    #     f_vec = reduce(
    #         vcat, [[f(out[i], p[i], t[j]) for j in axes(t, 2)] for i in axes(p, 2)])
    # end

    du = vec(dfdx(phi, x, θ))
    norm = prod(size(du))
    sum(abs2, du .- f_vec) / norm

end

#FourierNeuralOperator and Chain
function physics_loss(
        phi::PINOPhi{C, T}, prob::ODEProblem, x::Array, θ) where {C, T}
    p, t = x[1:(end - 1), :, :], x[[end], :, :]
    f = prob.f
    out = phi(x, θ)
    if size(out)[1] == 1
        out = dropdims(out, dims = 1)
    end

    fs = f.(vec(out), vec(p), vec(t))
    f_vec = vec(fs)
    # f_vec = reduce(
    #     vcat, [[f(out[i], p[i], t[j]) for j in axes(t, 2)] for i in axes(p, 2)])

    du = vec(dfdx(phi, x, θ))
    norm = prod(size(du))
    sum(abs2, du .- f_vec) / norm
end

function initial_condition_loss(
        phi::PINOPhi{C, T}, prob::ODEProblem, x, θ) where {
        C <: CompactLuxLayer{:DeepONet,}, T}
    p, t = x
    t0 = reshape([prob.tspan[1]], (1, 1, 1)) # t[:, [1], :]
    x0 = (p, t0) #TODO one time in get_trainset ?
    out = phi(x0, θ)
    u = vec(out)
    u0 = vec(fill(prob.u0, size(out)))
    norm = prod(size(u0))
    sum(abs2, u .- u0) / norm
end

#FourierNeuralOperator and Chain
function initial_condition_loss(
        phi::PINOPhi{C, T}, prob::ODEProblem, x, θ) where {C, T}
    p, t = x[1:end-1,:,:], x[[end],:,:]
    t0 = fill(prob.tspan[1], size(p))
    x0 = reduce(vcat, (p, t0)) #TODO one time in get_trainset

    out = phi(x0, θ)
    u = vec(out)
    u0 = vec(fill(prob.u0, size(out)))
    norm = prod(size(u0))
    sum(abs2, u .- u0) / norm
end

#TODO for input FourierNeuralOperator and Chain
function get_trainset(strategy::GridTraining, bounds, number_of_parameters, tspan)
    dt = strategy.dx
    p = collect([range(start = b[1], length = number_of_parameters, stop = b[2]) for b in bounds]...)
    # p = vcat([collect(reshape(p_i, 1, size(p_i, 1))) for p_i in p_]...)
    # t = collect(range(start = tspan[1], length = 1/dt, stop = tspan[2]))
    t = collect(tspan[1]:dt:tspan[2])
    # t = reshape(t_, 1, size(t_, 1), 1)
    # reduce(vcat, (p, t))
    # hcat((map(points -> collect(points), Iterators.product(p,t)))...)
    combinations = collect(Iterators.product(p, t))
    N = size(p, 1)
    M = size(t, 1)
    x = zeros(2, N, M)

    for i in 1:N
        for j in 1:M
            x[:, i, j] = [combinations[(i - 1) * M + j]...]
        end
    end
    x
end

function get_trainset(strategy::GridTraining, chain::CompactLuxLayer{:DeepONet,}, bounds,
        number_of_parameters, tspan)
    dt = strategy.dx
    # if size(bounds,1) == 1
    #     bound = bounds[1]
    #     p_ = range(start = bound[1], length = number_of_parameters, stop = bound[2])
    #     p = collect(reshape(p_, 1, size(p_,1)))
    # else
        p_ = [range(start = b[1], length = number_of_parameters, stop = b[2])
              for b in bounds]
        p = vcat([collect(reshape(p_i, 1, size(p_i,1))) for p_i in p_]...)
    # end

    t_ = collect(tspan[1]:dt:tspan[2])
    t = reshape(t_, 1, size(t_,1), 1)
    (p, t)
end

function get_trainset(strategy::StochasticTraining,
        chain::CompactLuxLayer{:DeepONet,}, bounds, number_of_parameters, tspan)
    # if size(bounds,1) == 1 #TODO reduce if ?
    #     bound = bounds[1]
    #     p = (bound[2] .- bound[1]) .* rand(1, number_of_parameters) .+ bound[1]
    # else
        p = reduce(vcat,
            [(bound[2] .- bound[1]) .* rand(1, number_of_parameters) .+ bound[1]
             for bound in bounds])
    # end
    t = (tspan[2] .- tspan[1]) .* rand(1, strategy.points,1) .+ tspan[1]
    (p, t)
end

function generate_loss(
        strategy::GridTraining, prob::ODEProblem, phi, bounds, number_of_parameters, tspan)
    x = get_trainset(strategy, bounds, number_of_parameters, tspan)
    function loss(θ, _)
        initial_condition_loss(phi, prob, x, θ) + physics_loss(phi, prob, x, θ)
    end
end

function generate_loss(
        strategy::StochasticTraining, prob::ODEProblem, phi, bounds, number_of_parameters, tspan)
    function loss(θ, _)
        x = get_trainset(strategy, bounds, number_of_parameters, tspan)
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

    if !(chain isa Lux.AbstractExplicitLayer)
        error("Only Lux.AbstractExplicitLayer neural networks are supported")

        if !(isa(chain, CompactLuxLayer{:DeepONet,}) ||
             chain.name == "FourierNeuralOperator") #TODO chain.name
            error("Only DeepONet and FourierNeuralOperator neural networks are supported with PINO ODE")
        end
    end

    phi, init_params = generate_pino_phi_θ(chain, init_params)

    isinplace(prob) &&
        throw(error("The PINOODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))

    try
        if chain isa CompactLuxLayer{:DeepONet,}
            in_dim = chain.layers.branch.layers.layer_1.in_dims
            u = rand(in_dim, number_of_parameters)
            v = rand(1, 10, 1)
            x = (u, v)
            phi(x, init_params)
        elseif chain.name == "FourierNeuralOperator"
            in_dim = chain.layers.lifting.in_dims
            v = rand(in_dim, number_of_parameters, 40)
            phi(v, θ)
        else #TODO identifier for simple Chain
            in_dim = chain.layers.layer_1.in_dims
            v = rand(in_dim, number_of_parameters, 40)
            phi(v, θ)
        end
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

    p, t = get_trainset(strategy, bounds, number_of_parameters, tspan)
    x = (p, t)
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
