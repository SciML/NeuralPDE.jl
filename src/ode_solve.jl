abstract type NeuralPDEAlgorithm <: SciMLBase.AbstractODEAlgorithm end

"""
    NNODE(chain, opt, init_params = nothing; strategy = nothing, autodiff = false,
        batch = true, param_estim = false, additional_loss = nothing,
        dataset = [], estim_collocate = false, kwargs...)

Algorithm for solving ordinary differential equations using a neural network. This is a
specialization of the physics-informed neural network which is used as a solver for a
standard `ODEProblem`.

!!! warning

    Note that NNODE only supports ODEs which are written in the out-of-place form, i.e.
    `du = f(u,p,t)`, and not `f(du,u,p,t)`. If not declared out-of-place, then the NNODE
    will exit with an error.

## Positional Arguments

* `chain`: A neural network architecture, defined as a `Lux.AbstractLuxLayer` or
           `Flux.Chain`. `Flux.Chain` will be converted to `Lux` using
           `adapt(FromFluxAdaptor(), chain)`.
* `opt`: The optimizer to train the neural network.
* `init_params`: The initial parameter of the neural network. By default, this is `nothing`
                 which thus uses the random initialization provided by the neural network
                 library.

## Keyword Arguments

* `additional_loss`: A function additional_loss(phi, θ) where phi are the neural network
                     trial solutions, θ are the weights of the neural network(s).
* `dataset`: Is either an empty Vector or a nested Vector of the form `[x̂, t, W]` where `x̂` are dependant variable observations, `t` are time points and `W` are quadrature weights for domain.
             The dataset is used to compute a L2 loss against the data and also for the Data Quadrature loss function.
             For multiple dependant variables, there will be multiple vectors with the last two vectors in dataset still being for `t`, `W`.
             Is empty by default assuming a forward problem is being solved.
* `autodiff`: The switch between automatic and numerical differentiation for
              the PDE operators. The reverse mode of the loss function is always
              automatic differentiation (via Zygote), this is only for the derivative
              in the loss function (the derivative with respect to time).
* `batch`: The batch size for the loss computation. Defaults to `true`, means the neural
           network is applied at a row vector of values `t` simultaneously, i.e. it's the
           batch size for the neural network evaluations. This requires a neural network
           compatible with batched data. `false` means which means the application of the
           neural network is done at individual time points one at a time. This is not
           applicable to `QuadratureTraining` where `batch` is passed in the `strategy`
           which is the number of points it can parallelly compute the integrand.
* `param_estim`: Boolean to indicate whether parameters of the differential equations are
                 learnt along with parameters of the neural network.
* `strategy`: The training strategy used to choose the points for the evaluations.
              Default of `nothing` means that `QuadratureTraining` with QuadGK is used if no
              `dt` is given, and `GridTraining` is used with `dt` if given.
* `estim_collocate`: A boolean value to indicate whether to use the Data Quadrature loss function or not. This is only relevant for ODE parameter estimation.
* `kwargs`: Extra keyword arguments are splatted to the Optimization.jl `solve` call.

## Examples

```julia
u0 = [1.0, 1.0]
ts = [t for t in 1:100]
(u_, t_) = (analytical_func(ts), ts)
function additional_loss(phi, θ)
    return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
end
alg = NNODE(chain, opt, additional_loss = additional_loss)
```

```julia
f(u,p,t) = cos(2pi*t)
tspan = (0.0, 1.0)
u0 = 0.0
prob = ODEProblem(linear, u0 ,tspan)
chain = Lux.Chain(Lux.Dense(1, 5, Lux.σ), Lux.Dense(5, 1))
opt = OptimizationOptimisers.Adam(0.1)
sol = solve(prob, NNODE(chain, opt), verbose = true, abstol = 1e-10, maxiters = 200)
```

## Solution Notes

Note that the solution is evaluated at fixed time points according to standard output
handlers such as `saveat` and `dt`. However, the neural network is a fully continuous
solution so `sol(t)` is an accurate interpolation (up to the neural network training
result). In addition, the `OptimizationSolution` is returned as `sol.k` for further
analysis.

## References

Lagaris, Isaac E., Aristidis Likas, and Dimitrios I. Fotiadis. "Artificial neural networks
for solving ordinary and partial differential equations." IEEE Transactions on Neural
Networks 9, no. 5 (1998): 987-1000.
"""
@concrete struct NNODE
    chain <: AbstractLuxLayer
    opt
    init_params
    autodiff::Bool
    batch
    strategy <: Union{Nothing, AbstractTrainingStrategy}
    param_estim
    additional_loss <: Union{Nothing, Function}
    dataset <: Union{Vector, Vector{<:Vector{<:AbstractFloat}}}
    estim_collocate::Bool
    kwargs
end

function NNODE(chain, opt, init_params = nothing; strategy = nothing, autodiff = false,
        batch = true, param_estim = false, additional_loss = nothing,
        dataset = [], estim_collocate = false, kwargs...)
    chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))
    return NNODE(chain, opt, init_params, autodiff, batch,
        strategy, param_estim, additional_loss, dataset, estim_collocate, kwargs)
end

"""
    ODEPhi(chain::Lux.AbstractLuxLayer, t, u0, st)

Internal struct, used for representing the ODE solution as a neural network in a form that
respects boundary conditions, i.e. `phi(t) = u0 + t*NN(t)`.
"""
@concrete struct ODEPhi
    u0
    t0
    smodel <: StatefulLuxLayer
end

function ODEPhi(model::AbstractLuxLayer, t0::Number, u0, st)
    return ODEPhi(u0, t0, StatefulLuxLayer{true}(model, nothing, st))
end

function generate_phi_θ(chain::AbstractLuxLayer, t, u0, ::Nothing)
    θ, st = LuxCore.setup(Random.default_rng(), chain)
    return ODEPhi(chain, t, u0, st), θ
end

function generate_phi_θ(chain::AbstractLuxLayer, t, u0, init_params)
    st = LuxCore.initialstates(Random.default_rng(), chain)
    return ODEPhi(chain, t, u0, st), init_params
end

function (f::ODEPhi)(t, θ)
    dev = safe_get_device(θ)
    return f(dev, safe_expand(dev, t), θ)
end

function (f::ODEPhi{<:Number})(dev, t::Number, θ)
    res = only(cdev(f.smodel(dev([t]), θ.depvar)))
    return f.u0 + (t - f.t0) * res
end

function (f::ODEPhi{<:Number})(_, t::AbstractVector, θ)
    return f.u0 .+ (t' .- f.t0) .* f.smodel(t', θ.depvar)
end

(f::ODEPhi)(dev, t::Number, θ) = dev(f.u0) .+ (t .- f.t0) .* f.smodel(dev([t]), θ.depvar)

(f::ODEPhi)(dev, t::AbstractVector, θ) = dev(f.u0) .+ (t' .- f.t0) .* f.smodel(t', θ.depvar)

"""
    ode_dfdx(phi, t, θ, autodiff)

Computes u' using either forward-mode automatic differentiation or numerical differentiation.
"""
function ode_dfdx end

function ode_dfdx(phi::ODEPhi, t, θ, autodiff::Bool)
    if autodiff
        t isa Number && return ForwardDiff.derivative(Base.Fix2(phi, θ), t)
        return ForwardDiff.jacobian(Base.Fix2(phi, θ), t)
    end
    ϵ = sqrt(eps(eltype(t)))
    return (phi(t .+ ϵ, θ) .- phi(t, θ)) ./ ϵ
end

"""
    inner_loss(phi, f, autodiff, t, θ, p, param_estim)

Simple L2 inner loss at a time `t` with parameters `θ` of the neural network.
"""
function inner_loss end

function inner_loss(phi::ODEPhi, f, autodiff::Bool, t::Number, θ, p, param_estim::Bool)
    p_ = param_estim ? θ.p : p
    return sum(abs2, ode_dfdx(phi, t, θ, autodiff) .- f(phi(t, θ), p_, t))
end

function inner_loss(
        phi::ODEPhi, f, autodiff::Bool, t::AbstractVector, θ, p, param_estim::Bool)
    p_ = param_estim ? θ.p : p
    out = phi(t, θ)
    fs = if phi.u0 isa Number
        reduce(hcat, [f(out[i], p_, tᵢ) for (i, tᵢ) in enumerate(t)])
    else
        reduce(hcat, [f(out[:, i], p_, tᵢ) for (i, tᵢ) in enumerate(t)])
    end
    dxdtguess = ode_dfdx(phi, t, θ, autodiff)
    return sum(abs2, fs .- dxdtguess) / length(t)
end

"""
    generate_loss(strategy, phi, f, autodiff, tspan, p, batch, param_estim)

Representation of the loss function, parametric on the training strategy `strategy`.
"""
function generate_loss(strategy::QuadratureTraining, phi, f, autodiff::Bool, tspan, p,
        batch, param_estim::Bool)
    integrand(t::Number, θ) = abs2(inner_loss(phi, f, autodiff, t, θ, p, param_estim))

    function integrand(ts, θ)
        return [abs2(inner_loss(phi, f, autodiff, t, θ, p, param_estim)) for t in ts]
    end

    function loss(θ, _)
        intf = BatchIntegralFunction(integrand, max_batch = strategy.batch)
        intprob = IntegralProblem(intf, (tspan[1], tspan[2]), θ)
        sol = solve(intprob, strategy.quadrature_alg; strategy.abstol,
            strategy.reltol, strategy.maxiters)
        return sol.u
    end

    return loss
end

function generate_loss(
        strategy::GridTraining, phi, f, autodiff::Bool, tspan, p, batch, param_estim::Bool)
    ts = tspan[1]:(strategy.dx):tspan[2]
    autodiff && throw(ArgumentError("autodiff not supported for GridTraining."))
    batch && return (θ, _) -> inner_loss(phi, f, autodiff, ts, θ, p, param_estim)
    return (θ, _) -> sum([inner_loss(phi, f, autodiff, t, θ, p, param_estim) for t in ts])
end

function generate_loss(strategy::StochasticTraining, phi, f, autodiff::Bool, tspan, p,
        batch, param_estim::Bool)
    autodiff && throw(ArgumentError("autodiff not supported for StochasticTraining."))
    return (θ, _) -> begin
        T = promote_type(eltype(tspan[1]), eltype(tspan[2]))
        ts = (tspan[2] - tspan[1]) .* rand(T, strategy.points) .+ tspan[1]
        if batch
            inner_loss(phi, f, autodiff, ts, θ, p, param_estim)
        else
            sum([inner_loss(phi, f, autodiff, t, θ, p, param_estim) for t in ts])
        end
    end
end

function generate_loss(
        strategy::WeightedIntervalTraining, phi, f, autodiff::Bool, tspan, p,
        batch, param_estim::Bool)
    autodiff && throw(ArgumentError("autodiff not supported for WeightedIntervalTraining."))
    minT, maxT = tspan
    weights = strategy.weights ./ sum(strategy.weights)
    N = length(weights)
    difference = (maxT - minT) / N

    ts = eltype(difference)[]
    for (index, item) in enumerate(weights)
        temp_data = rand(1, trunc(Int, strategy.points * item)) .* difference .+ minT .+
                    ((index - 1) * difference)
        append!(ts, temp_data)
    end

    batch && return (θ, _) -> inner_loss(phi, f, autodiff, ts, θ, p, param_estim)
    return (θ, _) -> sum([inner_loss(phi, f, autodiff, t, θ, p, param_estim) for t in ts])
end

function evaluate_tstops_loss(phi, f, autodiff::Bool, tstops, p, batch, param_estim::Bool)
    batch && return (θ, _) -> inner_loss(phi, f, autodiff, tstops, θ, p, param_estim)
    return (θ, _) -> sum([inner_loss(phi, f, autodiff, t, θ, p, param_estim)
                          for t in tstops])
end

function generate_loss(::QuasiRandomTraining, phi, f, autodiff::Bool, tspan)
    error("QuasiRandomTraining is not supported by NNODE since it's for high dimensional \
           spaces only. Use StochasticTraining instead.")
end

"""
L2 loss (needed for ODE parameter estimation).
"""
function generate_L2lossData(dataset, phi, n_output)
    isempty(dataset) && return 0
    return (θ, _) -> sum(sum(abs2, phi(dataset[end - 1], θ)[i, :] .- dataset[i])
    for i in 1:n_output)
end

"""
Data Quadrature loss function (provides very accurate solution, parameter estimates and a method for algorithmic sampling of a minimal set of data points for Inverse problems).
"""
function generate_L2loss2(f, autodiff, dataset, phi, n_output)
    isempty(dataset) && return 0
    t = dataset[end - 1]
    û = dataset[1:(end - 2)]
    quadrature_weights = dataset[end]

    function L2loss2(θ, _)
        nnsol = ode_dfdx(phi, t, θ, autodiff)
        ode_params = θ.p

        physsol = if n_output == 1
            [f(û[1][i], ode_params, tᵢ) for (i, tᵢ) in enumerate(t)]
        else
            [f([û[j][i] for j in 1:(length(dataset) - 2)], ode_params, tᵢ)
             for (i, tᵢ) in enumerate(t)]
        end
        # form of NN output matrix output dim x n
        deri_physsol = reduce(hcat, physsol)

        # Quadrature is applied on timewise losses
        # Gridtraining/trapezoidal rule quadrature_weights is dt.*ones(T, length(t))
        return sum(sum(abs2.(nnsol[i, :] .- deri_physsol[i, :]) .* quadrature_weights)
        for i in 1:n_output)
    end
end

@concrete struct NNODEInterpolation
    phi <: ODEPhi
    θ
end

(f::NNODEInterpolation)(t, ::Nothing, ::Type{Val{0}}, p, continuity) = f.phi(t, f.θ)
(f::NNODEInterpolation)(t, idxs, ::Type{Val{0}}, p, continuity) = f.phi(t, f.θ)[idxs]

function (f::NNODEInterpolation)(t::Vector, ::Nothing, ::Type{Val{0}}, p, continuity)
    out = f.phi(t, f.θ)
    return DiffEqArray([out[:, i] for i in axes(out, 2)], t)
end

function (f::NNODEInterpolation)(t::Vector, idxs, ::Type{Val{0}}, p, continuity)
    out = f.phi(t, f.θ)
    return DiffEqArray([out[idxs, i] for i in axes(out, 2)], t)
end

function (sol::ODESolution{T, N, U, U2, D, T2, R, D2, P, A})(
        t::AbstractVector{<:Number}, ::Type{deriv}, idxs::Nothing,
        continuity) where {T, N, U, U2, D, T2, R, D2, P, A <: NNODE, deriv}
    sol.interp(t, idxs, deriv, sol.prob.p, continuity)
end

SciMLBase.interp_summary(::NNODEInterpolation) = "Trained neural network interpolation"
SciMLBase.allowscomplex(::NNODE) = true

function SciMLBase.__solve(
        prob::SciMLBase.AbstractODEProblem,
        alg::NNODE,
        args...;
        dt = nothing,
        timeseries_errors = true,
        save_everystep = true,
        adaptive = false,
        abstol = 1.0f-6,
        reltol = 1.0f-3,
        verbose = false,
        saveat = nothing,
        maxiters = nothing,
        tstops = nothing
)
    (; u0, tspan, f, p) = prob
    t0 = tspan[1]
    # add estim_collocate, dataset (or nothing) in NNODE
    (; param_estim, estim_collocate, dataset, chain, opt, autodiff,
    init_params, batch, additional_loss, estim_collocate) = alg

    phi, init_params = generate_phi_θ(chain, t0, u0, init_params)

    (recursive_eltype(init_params) <: Complex && alg.strategy isa QuadratureTraining) &&
        error("QuadratureTraining cannot be used with complex parameters. Use other strategies.")

    init_params = if alg.param_estim
        ComponentArray(; depvar = init_params, p)
    else
        ComponentArray(; depvar = init_params)
    end

    @assert !isinplace(prob) "The NNODE solver only supports out-of-place ODE definitions, i.e. du=f(u,p,t)."

    strategy = if alg.strategy === nothing
        if dt !== nothing
            GridTraining(dt)
        else
            QuadratureTraining(; quadrature_alg = QuadGKJL(),
                reltol = convert(eltype(u0), reltol), abstol = convert(eltype(u0), abstol),
                maxiters, batch = 0)
        end
    else
        alg.strategy
    end

    inner_f = generate_loss(strategy, phi, f, autodiff, tspan, p, batch, param_estim)

    if !isempty(dataset) &&
       (length(dataset) < 3 || !(dataset isa Vector{<:Vector{<:AbstractFloat}}))
        error("Invalid dataset. The dataset would be a timeseries (x̂,t,W) with type: Vector{Vector{AbstractFloat}")
    end

    if isempty(dataset) && param_estim && isnothing(additional_loss)
        error("Dataset or an additional loss is required for Inverse problems performing Parameter Estimation.")
    elseif isempty(dataset) && estim_collocate
        error("Dataset is required for Inverse problems performing Parameter Estimation using the Data Quadrature loss function.")
    end

    n_output = length(u0)
    L2lossData = generate_L2lossData(dataset, phi, n_output)
    L2loss2 = generate_L2loss2(f, autodiff, dataset, phi, n_output)

    # Creates OptimizationFunction Object from total_loss
    function total_loss(θ, _)
        L2_loss = inner_f(θ, phi)

        if param_estim && estim_collocate
            L2_loss = L2_loss + L2lossData(θ, phi) + L2loss2(θ, phi)
        elseif param_estim && !isempty(dataset)
            L2_loss = L2_loss + L2lossData(θ, phi)
        end
        if additional_loss !== nothing
            L2_loss = L2_loss + additional_loss(phi, θ)
        end
        if tstops !== nothing
            num_tstops_points = length(tstops)
            tstops_loss_func = evaluate_tstops_loss(
                phi, f, autodiff, tstops, p, batch, param_estim)
            tstops_loss = tstops_loss_func(θ, phi)
            if strategy isa GridTraining
                num_original_points = length(tspan[1]:(strategy.dx):tspan[2])
            elseif strategy isa Union{WeightedIntervalTraining, StochasticTraining}
                num_original_points = strategy.points
            else
                return L2_loss + tstops_loss
            end
            total_original_loss = L2_loss * num_original_points
            total_tstops_loss = tstops_loss * num_tstops_points
            total_points = num_original_points + num_tstops_points
            L2_loss = (total_original_loss + total_tstops_loss) / total_points
        end
        return L2_loss
    end

    opt_algo = ifelse(strategy isa QuadratureTraining, AutoForwardDiff(), AutoZygote())
    optf = OptimizationFunction(total_loss, opt_algo)

    plen = maxiters === nothing ? 6 : ndigits(maxiters)
    callback = function (p, l)
        if verbose
            if maxiters === nothing
                @printf("[NNODE]\tIter: [%*d]\tLoss: %g\n", plen, p.iter, l)
            else
                @printf("[NNODE]\tIter: [%*d/%d]\tLoss: %g\n", plen, p.iter, maxiters, l)
            end
        end
        return l < abstol
    end

    optprob = OptimizationProblem(optf, init_params)
    res = solve(optprob, opt; callback, maxiters, alg.kwargs...)

    #solutions at timepoints
    if saveat isa Number
        ts = tspan[1]:saveat:tspan[2]
    elseif saveat isa AbstractArray
        ts = saveat
    elseif dt !== nothing
        ts = tspan[1]:dt:tspan[2]
    elseif save_everystep
        ts = range(tspan[1], tspan[2], length = 100)
    else
        ts = [tspan[1], tspan[2]]
    end

    if u0 isa Number
        u = [first(phi(t, res.u)) for t in ts]
    else
        u = [phi(t, res.u) for t in ts]
    end

    sol = SciMLBase.build_solution(prob, alg, ts, u; k = res, dense = true,
        interp = NNODEInterpolation(phi, res.u), calculate_error = false,
        retcode = ReturnCode.Success, original = res, resid = res.objective)

    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(
            sol; timeseries_errors = true, dense_errors = false)

    return sol
end
