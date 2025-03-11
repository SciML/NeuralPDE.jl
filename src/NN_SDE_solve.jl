@concrete struct NNSDE
    chain <: AbstractLuxLayer
    opt
    init_params
    autodiff::Bool
    batch
    strategy <: Union{Nothing, AbstractTrainingStrategy}
    param_estim::Bool
    additional_loss <: Union{Nothing, Function}
    sub_batch::Number
    numensemble::Number
    kwargs
end

function NNSDE(chain, opt, init_params = nothing; strategy = nothing, autodiff = false,
        batch = true, param_estim = false, additional_loss = nothing,
        sub_batch = 1, numensemble = 10, kwargs...)
    chain isa AbstractLuxLayer || (chain = FromFluxAdaptor()(chain))
    return NNSDE(chain, opt, init_params, autodiff, batch,
        strategy, param_estim, additional_loss, sub_batch, numensemble, kwargs)
end

"""
    SDEPhi(chain::Lux.AbstractLuxLayer, t, u0, st)

Internal struct, used for representing the SDE solution as a neural network in a form that
respects boundary conditions, i.e. `phi(inp) = u0 + inp[1]*NN(inp)`.
"""
@concrete struct SDEPhi
    u0
    t0
    smodel <: StatefulLuxLayer
end

function SDEPhi(model::AbstractLuxLayer, t0::Number, u0, st)
    return SDEPhi(u0, t0, StatefulLuxLayer{true}(model, nothing, st))
end

function (f::SDEPhi)(inp, θ)
    dev = safe_get_device(θ)
    return f(dev, safe_expand(dev, inp), θ)
end

function (f::SDEPhi{<:Number})(dev, inp::Array{<:Number, 1}, θ)
    res = only(cdev(f.smodel(dev(inp), θ.depvar)))
    return f.u0 + (inp[1] - f.t0) * res
end

function (f::SDEPhi)(dev, inp::Array{<:Number, 1}, θ)
    return dev(f.u0) .+ (inp[1] - f.t0) .* f.smodel(dev(inp), θ.depvar)
end

function generate_phi(chain::AbstractLuxLayer, t, u0, ::Nothing)
    θ, st = LuxCore.setup(Random.default_rng(), chain)
    return SDEPhi(chain, t, u0, st), θ
end

function generate_phi(chain::AbstractLuxLayer, t, u0, init_params)
    st = LuxCore.initialstates(Random.default_rng(), chain)
    return SDEPhi(chain, t, u0, st), init_params
end

"""
    ∂u_∂t(phi, inp, θ, autodiff)

Computes for sde's solution u, u' using either forward-mode automatic differentiation or numerical differentiation.
"""
function ∂u_∂t end

# earlier input as number or abstract vector, now becomes array1 or matrix
# input first col must be time, rest all z_i's
# vector of t_i's sub_batch input vectors = inputs
# returns a vector of gradients for each sub_batch, for each method call over all its sub_batches
function ∂u_∂t(phi::SDEPhi, inputs::Array{<:Array{<:Number, 1}}, θ, autodiff::Bool)
    autodiff &&
        return [ForwardDiff.gradient(
                    t -> (phi(vcat(t, input[2:end]), θ)), input[1])
                for input in inputs]
    ϵ = sqrt(eps(eltype(inputs[1])))
    return [(phi(vcat(input[1] + ϵ, input[2:end]), θ) .- phi(input, θ)) ./ ϵ
            for input in inputs]
end

"""
    inner_sde_loss(phi, f, autodiff, t, θ, p, param_estim)

Simple L2 inner loss for the SDE at a time `t` and random variables z_i with parameters `θ` of the neural network.
    
For NNSE instead of a matrix, input is a N x n_sub x (t+n_z) dim array N -> n timepoints, n_sub -> n_sub_batch, t+n_z -> chain/phi input dims.
Inner Sde loss enforces strong solution across sub_batches (strong sol convergence implies weak sol convergence but not vice versa)
Note: NNODE, NNSDE take only a single Neural Network which is multioutput or singleoutput

"""
function inner_sde_loss end

# no batching
function inner_sde_loss(
        phi::SDEPhi, f, g, autodiff::Bool, inputs::Array{<:Array{<:Number, 1}},
        θ, p, param_estim::Bool, n_sub_batch)
    p_ = param_estim ? θ.p : p

    # phi's many outputs for the same timepoint's many sub_batches
    # must also cover multioutput case
    u = [phi(sub_batch_input, θ) for sub_batch_input in inputs]

    # for t's sub_batch we consider the i_th batch in sub_batch as indvar, z_i... inputs for the ith phi output (also covers sub_batch=1 case)
    fs = if phi.u0 isa Number
        [f(u[i][1], p_, inp[1]) +
         g(u[i][1], p_, inp[1]) * √2 *
         sum(inp[1 + j] * cos((j - 1 / 2)pi * inp[1]) for j in 1:(length(inp) - 1))
         for (i, inp) in enumerate(inputs)]
    else
        # will be vector in multioutput case
        [f(u[i], p_, inp[1]) +
         g(u[i], p_, inp[1]) * √2 *
         sum(inp[1 + j] * cos((j - 1 / 2)pi * inp[1]) for j in 1:(length(inp) - 1))
         for (i, inp) in enumerate(inputs)]
    end

    # gradient at t is not affected by sub_batch z_i values beyond use in phi(t+ϵ)
    dudt = ∂u_∂t(phi, inputs, θ, autodiff)

    # initial .- broadcasting over  NN multiple/single outputs and subbatches simultaneously
    # fs and dudt size is (n_sub_batch, NN_output_size), loss is for one timepoint
    # broadcasted sum(abs2,) over losses for each sub_batch, where rows - sub_batch's and cols - NN multiple/single outputs
    # finally sum over vector of L2 errors (all the sub_batches) (direct sum/mean of L2's across sub_batches)
    return sum(sum.(abs2, fs .- dudt))
end

# batching case
function inner_sde_loss(
        phi::SDEPhi, f, g, autodiff::Bool, inputs::Array{<:Array{<:Array{<:Number, 1}}},
        θ, p, param_estim::Bool, n_sub_batch)
    p_ = param_estim ? θ.p : p

    # phi's many outputs for each timepoint's many sub_batches, input for each t_i, sub_batch_input for each input's sub_batches
    # must also cover multioutput case
    u = [[phi(sub_batch_input, θ) for sub_batch_input in input] for input in inputs]

    # for each t_i's sub_batch we consider the i_th batch in sub_batch as indvar, z_i... inputs for the i_th phi output (covers subbatch=1 case for each t_i)
    fs = if phi.u0 isa Number
        [[f(u[i][j][1], p_, inpi[1]) +
          g(u[i][j][1], p_, inpi[1]) * 2^(1 / 2) *
          sum(inpi[1 + k] * cos((k - 1 / 2)pi * inpi[1]) for k in 1:(length(inpi) - 1))
          for (j, inpi) in enumerate(inp)] for (i, inp) in enumerate(inputs)]
    else
        [[f(u[i][j], p_, inpi[1]) +
          g(u[i][j], p_, inpi[1]) * 2^(1 / 2) *
          sum(inpi[1 + k] * cos((k - 1 / 2)pi * inpi[1]) for k in 1:(length(inpi) - 1))
          for (j, inpi) in enumerate(inp)] for (i, inp) in enumerate(inputs)]
    end

    # fs[i] is made of n=n_sub_batch, vectors of n=n_output_dims dim each
    # gradient at t_i's is not affected by their sub_batch's z_i values beyond use in phi(ti+ϵ)
    dudt = [∂u_∂t(phi, inpi, θ, autodiff) for inpi in inputs]

    # Taking MSE across Z, each fs and du/dt has n_sub_batch elements in them
    # sum used for each timepoint's sub_batch as strong convergence enforced for each WienerProcess realization
    # same explanation as in non batching case, final mean L2 aggregates over all timepoints.
    return sum(sum(sum.(abs2, fs[i] .- dudt[i])) for i in eachindex(inputs)) /
           length(inputs)
end

"""
    add_rand_coeff(times, n_z)
n_z is the number of orthogonal basis (probability space) of Random variables taken in the KKl expansion for an SDE.
n_z can also be a list of sampled values
returns a list appending n_z or n = n_z sampled (Uniform Gaussian) random variables values to a fixed time's value or a list of times.
"""
function add_rand_coeff(times, n_z::Number)
    times isa Number && return vcat(times, rand(Normal(0, 1), n_z))
    return [vcat(time, rand(Normal(0, 1), n_z))
            for time in times]
end

"""
    generate_loss(strategy, phi, f, autodiff, tspan, p, batch, param_estim)

Representation of the loss function, parametric on the training strategy `strategy`.
"""
function generate_loss(
        strategy::QuadratureTraining, phi, f, g, autodiff::Bool, tspan, n_z, n_sub_batch, p,
        batch, param_estim::Bool)
    inputs = AbstractVector{Any}[]
    function integrand(t::Number, θ)
        inputs = [[add_rand_coeff(t, n_z) for i in 1:n_sub_batch]]
        return abs2(inner_sde_loss(
            phi, f, g, autodiff, inputs, θ, p, param_estim, n_sub_batch))
    end

    # when ts is a list
    function integrand(ts, θ)
        inputs = [[add_rand_coeff(t, n_z) for i in 1:n_sub_batch] for t in ts]
        return [abs2(inner_sde_loss(
                    phi, f, g, autodiff, input, θ, p, param_estim, n_sub_batch))
                for input in inputs]
    end

    function loss(θ, _)
        intf = BatchIntegralFunction(integrand, max_batch = strategy.batch)
        intprob = IntegralProblem(intf, (tspan[1], tspan[2]), θ)
        sol = solve(intprob, strategy.quadrature_alg; strategy.abstol,
            strategy.reltol, strategy.maxiters)
        return sol.u
    end

    return loss, inputs
end

function generate_loss(
        strategy::GridTraining, phi, f, g, autodiff::Bool, tspan, n_z, n_sub_batch, p, batch, param_estim::Bool)
    ts = tspan[1]:(strategy.dx):tspan[2]
    # in (t,n_i,..) space we solve at one point, NN(input) can also represent only this point if subbatch=1
    # inp = add_rand_coeff(ts, n_z)
    # for each ti in t we have n=n_sub_batch phi onput possibilities
    inputs = [[add_rand_coeff(t, n_z) for i in 1:n_sub_batch] for t in ts]

    autodiff && throw(ArgumentError("autodiff not supported for GridTraining."))
    batch &&
        return (θ, _) -> inner_sde_loss(
            phi, f, g, autodiff, inputs, θ, p, param_estim, n_sub_batch),
        inputs
    return (θ, _) -> sum([inner_sde_loss(
                              phi, f, g, autodiff, input, θ, p, param_estim, n_sub_batch)
                          for input in inputs]),
    inputs
end

function generate_loss(strategy::StochasticTraining, phi, f, g, autodiff::Bool,
        tspan, n_z, n_sub_batch, p, batch, param_estim::Bool)
    autodiff && throw(ArgumentError("autodiff not supported for StochasticTraining."))
    inputs = AbstractVector{Any}[]

    return (θ, _) -> begin
        T = promote_type(eltype(tspan[1]), eltype(tspan[2]))
        ts = ((tspan[2] - tspan[1]) .* rand(T, strategy.points) .+ tspan[1])
        inputs = [[add_rand_coeff(t, n_z) for i in 1:n_sub_batch] for t in ts]

        if batch
            inner_sde_loss(phi, f, g, autodiff, inputs, θ, p, param_estim, n_sub_batch)
        else
            sum([inner_sde_loss(phi, f, g, autodiff, input, θ, p, param_estim, n_sub_batch)
                 for input in inputs])
        end
    end,
    inputs
end

function generate_loss(
        strategy::WeightedIntervalTraining, phi, f, g, autodiff::Bool, tspan, n_z, n_sub_batch, p,
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
    inputs = [[add_rand_coeff(t, n_z) for i in 1:n_sub_batch] for t in ts]

    batch &&
        return (θ, _) -> inner_sde_loss(
            phi, f, g, autodiff, inputs, θ, p, param_estim, n_sub_batch),
        inputs
    return (θ, _) -> sum([inner_sde_loss(
                              phi, f, g, autodiff, input, θ, p, param_estim, n_sub_batch)
                          for input in inputs]),
    inputs
end

function evaluate_tstops_loss(
        phi, f, g, autodiff::Bool, tstops, n_z, n_sub_batch, p, batch, param_estim::Bool)
    inputs = [[add_rand_coeff(t, n_z) for i in 1:n_sub_batch] for t in tstops]
    batch &&
        return (θ, _) -> inner_sde_loss(
            phi, f, g, autodiff, inputs, θ, p, param_estim, n_sub_batch),
        inputs
    return (θ, _) -> sum([inner_sde_loss(
                              phi, f, g, autodiff, input, θ, p, param_estim, n_sub_batch)
                          for input in inputs]),
    inputs
end

function generate_loss(::QuasiRandomTraining, phi, f, g, autodiff::Bool,
        tspan, n_z, n_sub_batch, p, batch, param_estim::Bool)
    error("QuasiRandomTraining is not supported by NNODE since it's for high dimensional \
           spaces only. Use StochasticTraining instead.")
end

@concrete struct NNSDEInterpolation
    phi <: SDEPhi
    θ
end

(f::NNSDEInterpolation)(inp, ::Nothing, ::Type{Val{0}}, p, continuity) = f.phi(inp, f.θ)
(f::NNSDEInterpolation)(inp, idxs, ::Type{Val{0}}, p, continuity) = f.phi(inp, f.θ)[idxs]

function (f::NNSDEInterpolation)(
        inp::Array{<:Number, 1}, ::Nothing, ::Type{Val{0}}, p, continuity)
    out = f.phi(inp, f.θ)
    return DiffEqArray([out[:, i] for i in axes(out, 2)], inp)
end

function (f::NNSDEInterpolation)(
        inp::Array{<:Number, 1}, idxs, ::Type{Val{0}}, p, continuity)
    out = f.phi(inp, f.θ)
    return DiffEqArray([out[idxs, i] for i in axes(out, 2)], inp)
end

SciMLBase.interp_summary(::NNSDEInterpolation) = "Trained neural network interpolation"
SciMLBase.allowscomplex(::NNSDE) = true

@concrete struct SDEsol
    solution
    mean_fit::AbstractVector
    timepoints::AbstractVector{<:Number}
    ensemble_fits::AbstractVector
    ensemble_inputs::AbstractVector
    numensemble::Number
    training_sets::AbstractVector
end

function SciMLBase.__solve(
        prob::SciMLBase.AbstractSDEProblem,
        alg::NNSDE,
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
    (; u0, tspan, f, g, p) = prob
    # rescaling timespan and discretization so KKL expansion can be applied for loss formulation
    dt = dt / abs(tspan[2] - tspan[1])
    tspan = tspan ./ tspan[end]

    t0 = tspan[1]
    (; param_estim, chain, opt, autodiff, init_params, batch, additional_loss, sub_batch, numensemble) = alg
    n_z = chain[1].in_dims - 1

    phi, init_params = generate_phi(chain, t0, u0, init_params)

    (recursive_eltype(init_params) <: Complex && alg.strategy isa QuadratureTraining) &&
        error("QuadratureTraining cannot be used with complex parameters. Use other strategies.")

    init_params = if alg.param_estim
        ComponentArray(; depvar = init_params, p)
    else
        ComponentArray(; depvar = init_params)
    end

    @assert !isinplace(prob) "The NNSDE solver only supports out-of-place SDE definitions, i.e. du=f(u,p,t) + g(u,p,t)*dW(t)"

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

    inner_f, training_sets = generate_loss(
        strategy, phi, f, g, autodiff, tspan, n_z, sub_batch, p, batch, param_estim)

    (param_estim && additional_loss === nothing) &&
        throw(ArgumentError("Please provide `additional_loss` in `NNSDE` for parameter estimation (`param_estim` is true)."))

    # Creates OptimizationFunction Object from total_loss
    function total_loss(θ, _)
        L2_loss = inner_f(θ, phi)
        if additional_loss !== nothing
            L2_loss = L2_loss + additional_loss(phi, θ)
        end
        if tstops !== nothing
            num_tstops_points = length(tstops)
            tstops_loss_func = evaluate_tstops_loss(
                phi, f, g, autodiff, tstops, n_z, sub_batch, p, batch, param_estim)
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
                @printf("[NNSDE]\tIter: [%*d]\tLoss: %g\n", plen, p.iter, l)
            else
                @printf("[NNSDE]\tIter: [%*d/%d]\tLoss: %g\n", plen, p.iter, maxiters, l)
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
    ts = collect(ts)

    ensembles = []
    ensemble_inputs = []
    for i in 1:numensemble
        inputs = add_rand_coeff(ts, n_z)

        if u0 isa Number
            u = [(u0 + (input[1] - t0) * first(phi(input, res.u)))
                 for input in inputs]
        else
            u = [(u0 .+ (input[1] - t0) * phi(input, res.u)) for input in inputs]
        end
        push!(ensembles, u)
        push!(ensemble_inputs, inputs)
    end
    sde_sols = hcat(ensembles...)
    mean_sde_sol = [mean(sde_sols[i, :]) for i in 1:size(ts)[1]]

    sol = SciMLBase.build_solution(prob, alg, ts, mean_sde_sol; k = res, dense = true,
        interp = NNSDEInterpolation(phi, res.u), calculate_error = false,
        retcode = ReturnCode.Success, original = res, resid = res.objective)

    SciMLBase.has_analytic(prob.f) &&
        SciMLBase.calculate_solution_errors!(
            sol; timeseries_errors = true, dense_errors = false)

    return SDEsol(
        sol, mean_sde_sol, ts, ensembles, ensemble_inputs, numensemble, training_sets)
end
