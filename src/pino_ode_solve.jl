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

* `chain`: A neural network architecture, defined as a `AbstractLuxLayer` or `Flux.Chain`.
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
@concrete struct PINOODE
    chain
    opt
    bounds
    number_of_parameters::Int
    init_params
    strategy <: Union{Nothing, AbstractTrainingStrategy}
    additional_loss <: Union{Nothing, Function}
    kwargs
end

function PINOODE(chain,
        opt,
        bounds,
        number_of_parameters;
        init_params = nothing,
        strategy = nothing,
        additional_loss = nothing,
        kwargs...)
    chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))
    return PINOODE(chain, opt, bounds, number_of_parameters,
        init_params, strategy, additional_loss, kwargs)
end

@concrete struct PINOPhi
    model <: AbstractLuxLayer
    smodel <: StatefulLuxLayer
end

function PINOPhi(model::AbstractLuxLayer, st)
    return PINOPhi(model, StatefulLuxLayer{false}(model, nothing, st))
end

function generate_pino_phi_θ(chain::AbstractLuxLayer, ::Nothing)
    θ, st = LuxCore.setup(Random.default_rng(), chain)
    PINOPhi(chain, st), θ
end

function generate_pino_phi_θ(chain::AbstractLuxLayer, init_params)
    st = LuxCore.initialstates(Random.default_rng(), chain)
    PINOPhi(chain, st), init_params
end

function (f::PINOPhi{C, T})(x, θ) where {C <: AbstractLuxLayer, T}
    dev = safe_get_device(θ)
    return f(dev, safe_expand(dev, x), θ)
end

function (f::PINOPhi{C, T})(dev, x, θ) where {C <: AbstractLuxLayer, T}
    f.smodel(dev(x), θ)
end

function dfdx(phi::PINOPhi{C, T}, x::Tuple, θ) where {C <: DeepONet, T}
    p, t = x
    branch_left, branch_right = p, p
    trunk_left, trunk_right = t .+ sqrt(eps(eltype(t))), t
    x_left = (branch_left, trunk_left)
    x_right = (branch_right, trunk_right)
    (phi(x_left, θ) .- phi(x_right, θ)) ./ sqrt(eps(eltype(t)))
end

function dfdx(phi::PINOPhi{C, T}, x::Array,
        θ) where {C <: Lux.Chain, T}
    ε = [zeros(eltype(x), size(x)[1] - 1)..., sqrt(eps(eltype(x)))]
    (phi(x .+ ε, θ) - phi(x, θ)) ./ sqrt(eps(eltype(x)))
end

function physics_loss(
        phi::PINOPhi{C, T}, prob::ODEProblem, x::Tuple, θ) where {C <: DeepONet, T}
    p, t = x
    f = prob.f
    out = phi(x, θ)
    if size(p, 1) == 1
        f_vec = reduce(hcat,
            [reduce(vcat, [f(out[j, i], p[1, i], t[j]) for j in axes(t, 2)])
             for i in axes(p, 2)])
    else
        f_vec = reduce(hcat,
            [reduce(vcat, [f(out[j, i], p[:, i], t[j]) for j in axes(t, 2)])
             for i in axes(p, 2)])
    end
    du = dfdx(phi, x, θ)
    norm = prod(size(du))
    sum(abs2, du .- f_vec) / norm
end

function physics_loss(
        phi::PINOPhi{C, T}, prob::ODEProblem, x::Tuple, θ) where {
        C <: Lux.Chain, T}
    p, t = x
    x_ = reduce(vcat, x)
    f = prob.f
    out = phi(x_, θ)
    if size(p, 1) == 1 && size(out, 1) == 1
        f_vec = f.(out, p, t)
    elseif size(p, 1) > 1
        f_vec = reduce(hcat,
            [reduce(vcat, [f(out[1, i, j], p[:, i, j], t[1, i, j]) for j in axes(t, 3)])
             for i in axes(p, 2)])
    elseif size(out, 1) > 1
        f_vec = reduce(hcat,
            [reduce(vcat, [f(out[:, i, j], p[1, i, j], t[1, i, j]) for j in axes(t, 3)])
             for i in axes(p, 2)])
    end
    du = dfdx(phi, x_, θ)
    norm = prod(size(out))
    sum(abs2, du .- f_vec) / norm
end

function initial_condition_loss(
        phi::PINOPhi{C, T}, prob::ODEProblem, x, θ) where {
        C <: DeepONet, T}
    p, t = x
    t0 = reshape([prob.tspan[1]], (1, 1, 1))
    x0 = (p, t0)
    u = phi(x0, θ)
    u0 = size(prob.u0, 1) == 1 ? fill(prob.u0, size(u)) :
         reduce(vcat, [fill(u0, size(u)) for u0 in prob.u0])
    norm = prod(size(u0))
    sum(abs2, u .- u0) / norm
end

function initial_condition_loss(
        phi::PINOPhi{C, T}, prob::ODEProblem, x::Tuple, θ) where {
        C <: Lux.Chain, T}
    p, t = x
    t0 = fill(prob.tspan[1], size(t))
    x0 = reduce(vcat, (p, t0))
    u = phi(x0, θ)
    u0 = size(prob.u0, 1) == 1 ? fill(prob.u0, size(t)) :
         reduce(vcat, [fill(u0, size(t)) for u0 in prob.u0])
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
        strategy::GridTraining, chain::Chain, bounds, number_of_parameters, tspan)
    dt = strategy.dx
    tspan_ = tspan[1]:dt:tspan[2]
    pspan = [range(start = b[1], length = number_of_parameters, stop = b[2])
             for b in bounds]
    x_ = hcat(vec(map(
        points -> collect(points), Iterators.product([pspan..., tspan_]...)))...)
    x = reshape(x_, size(bounds, 1) + 1, prod(size.(pspan, 1)), size(tspan_, 1))
    p, t = x[1:(end - 1), :, :], x[[end], :, :]
    (p, t)
end

function get_trainset(
        strategy::StochasticTraining, chain::Union{DeepONet, Chain},
        bounds, number_of_parameters, tspan)
    (number_of_parameters != strategy.points && chain isa Chain) &&
        throw(error("number_of_parameters should be the same strategy.points for StochasticTraining"))
    p = reduce(vcat,
        [(bound[2] .- bound[1]) .* rand(1, number_of_parameters) .+ bound[1]
         for bound in bounds])
    t = (tspan[2] .- tspan[1]) .* rand(1, strategy.points, 1) .+ tspan[1]
    (p, t)
end

function generate_loss(
        strategy::GridTraining, prob::ODEProblem, phi, bounds, number_of_parameters, tspan)
    x = get_trainset(strategy, phi.smodel.model, bounds, number_of_parameters, tspan)
    function loss(θ, _)
        initial_condition_loss(phi, prob, x, θ) + physics_loss(phi, prob, x, θ)
    end
end

function generate_loss(
        strategy::StochasticTraining, prob::ODEProblem, phi, bounds, number_of_parameters, tspan)
    function loss(θ, _)
        x = get_trainset(strategy, phi.smodel.model, bounds, number_of_parameters, tspan)
        initial_condition_loss(phi, prob, x, θ) + physics_loss(phi, prob, x, θ)
    end
end

"""
PINOODEInterpolation(phi, θ)

Interpolation of the solution of the ODE using a trained neural network.

## Arguments
* `phi`: The neural network
* `θ`: The parameters of the neural network.
```
"""
@concrete struct PINOODEInterpolation{T <: PINOPhi, T2}
    phi::T
    θ::T2
end

"""
Override interpolation method for PINOODEInterpolation

## Arguments
* `x`: Input data on which the solution is to be interpolated.
## Example

```jldoctest
interp = PINOODEInterpolation(phi, θ)
x = rand(2, 50, 10)
interp(x)
```
"""
(f::PINOODEInterpolation)(x::AbstractArray) = f.phi(x, f.θ)

"""
Override interpolation method for PINOODEInterpolation

## Arguments
# * `p`: The parameters points on which the solution is to be interpolated.
# * `t`: The time points on which the solution is to be interpolated.

## Example
```jldoctest
interp = PINOODEInterpolation(phi, θ)
p,t = rand(1, 50, 10), rand(1, 50, 10)
interp(p, t)
```
"""
(f::PINOODEInterpolation)(p, t) = f(t, nothing, Val{0}, p, nothing)

function (f::PINOODEInterpolation)(
        t::AbstractArray, ::Nothing, ::Type{Val{0}}, p::AbstractArray, continuity)
    if f.phi.model isa DeepONet
        f.phi((p, t), f.θ)
    elseif f.phi.model isa Chain
        if size(p, 2) != size(t, 2)
            error("t should be same size as p")
        end
        f.phi(reduce(vcat, (p, t)), f.θ)
    else
        error("Only DeepONet and Chain neural networks are supported with PINO ODE")
    end
end

function (f::PINOODEInterpolation)(
        t::Number, ::Nothing, ::Type{Val{0}}, p::AbstractArray, continuity)
    if f.phi.model isa DeepONet
        t_ = [t]
        f.phi((p, t_), f.θ)
    elseif f.phi.model isa Chain
        t_ = fill(t, size(p))
        f.phi(reduce(vcat, (p, t_)), f.θ)
    else
        error("Only DeepONet and Chain neural networks are supported with PINO ODE")
    end
end

SciMLBase.interp_summary(::PINOODEInterpolation) = "Trained neural network interpolation"
SciMLBase.allowscomplex(::PINOODE) = true

function (sol::AbstractODESolution)(
        t::AbstractArray, ::Type{deriv}, idxs::Nothing,
        continuity) where {deriv}
    sol.interp(t, idxs, deriv, sol.prob.p, continuity)
end

function SciMLBase.__solve(prob::SciMLBase.AbstractODEProblem,
        alg::PINOODE,
        args...;
        abstol = 1.0f-8,
        reltol = 1.0f-3,
        verbose = false,
        saveat = nothing,
        maxiters = nothing)
    (; tspan, u0, f) = prob
    (; chain, opt, bounds, number_of_parameters, init_params, strategy, additional_loss) = alg

    if !(chain isa AbstractLuxLayer)
        error("Only Lux.AbstractLuxLayer neural networks are supported")

        if !(chain isa DeepONet) || !(chain isa Chain)
            error("Only DeepONet and Chain neural networks are supported with PINO ODE")
        end
    end

    phi, init_params = generate_pino_phi_θ(chain, init_params)

    init_params = ComponentArray(init_params)

    isinplace(prob) &&
        throw(error("The PINOODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."))

    try
        if chain isa DeepONet
            in_dim = chain.branch.layers.layer_1.in_dims
            u = rand(in_dim, number_of_parameters)
            v = rand(1, 10, 1)
            x = (u, v)
            phi(x, init_params)
        end
        if chain isa Chain
            in_dim = chain.layers.layer_1.in_dims
            x = rand(in_dim, number_of_parameters)
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

    inner_f = generate_loss(
        strategy, prob, phi, bounds, number_of_parameters, tspan)

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

    (p, t) = get_trainset(strategy, phi.smodel.model, bounds, number_of_parameters, tspan)
    interp = PINOODEInterpolation(phi, res.u)
    u = interp(p, t)
    prob_sol = ODEProblem(f.f, u0, tspan, p)

    sol = SciMLBase.build_solution(prob_sol, alg, t, u;
        k = res, dense = true,
        interp = interp,
        calculate_error = false,
        retcode = ReturnCode.Success,
        original = res,
        resid = res.objective)
    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(sol; timeseries_errors = true,
            dense_errors = false)
    sol
end
