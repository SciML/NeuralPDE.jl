"""
    GridTraining(dx)

A training strategy that uses the grid points in a multidimensional grid
with spacings `dx`. If the grid is multidimensional, then `dx` is expected
to be an array of `dx` values matching the dimension of the domain,
corresponding to the grid spacing in each dimension.

## Positional Arguments

* `dx`: the discretization of the grid.
"""
@concrete struct GridTraining <: AbstractTrainingStrategy
    dx
end

function get_dataset_train_points(eqs, train_sets, pinnrep)
    dict_depvar_input = pinnrep.dict_depvar_input
    depvars = pinnrep.depvars
    dict_depvars = pinnrep.dict_depvars
    dict_indvars = pinnrep.dict_indvars

    symbols_input = [(i, dict_depvar_input[i]) for i in depvars]
    # [(:u, [:t])]
    eq_args = NeuralPDE.get_argument(eqs, dict_indvars, dict_depvars)
    # [[:t]]

    points = [vcat([train_sets[i][:, 2:end]'
                    for i in eachindex(symbols_input) if symbols_input[i][2] == eq_arg]...)
              for eq_arg in eq_args]

    return points
end

# include dataset points in pde_residual loglikelihood (BayesianPINN)
function merge_strategy_with_loglikelihood_function(pinnrep::PINNRepresentation,
        strategy::GridTraining, datafree_pde_loss_function,
        datafree_bc_loss_function; train_sets_pde = nothing, train_sets_bc = nothing)
    eltypeθ = recursive_eltype(pinnrep.flat_init_params)
    adaptor = EltypeAdaptor{eltypeθ}()
        strategy::GridTraining,
        datafree_pde_loss_function,
        datafree_bc_loss_function; train_sets_pde = nothing, train_sets_bc=nothing)
    @unpack domains, eqs, bcs, dict_indvars, dict_depvars, flat_init_params = pinnrep

    eltypeθ = eltype(pinnrep.flat_init_params)

    # is vec as later each _set in pde_train_sets are columns as points transformed to vector of points (pde_train_sets must be rowwise)
    pde_loss_functions = if !(train_sets_pde isa Nothing)
        pde_train_sets = adapt.(parameterless_type(ComponentArrays.getdata(flat_init_params)),
            train_sets_pde)

        [get_loss_function(_loss, _set, eltypeθ, strategy)
                              for (_loss, _set) in zip(datafree_pde_loss_function,
            pde_train_sets)]
    else
        nothing
    end
    
    bc_loss_functions = if !(train_sets_bc isa Nothing)
        bcs_train_sets = adapt.(parameterless_type(ComponentArrays.getdata(flat_init_params)),
            train_sets_bc)

        [get_loss_function(_loss, _set, eltypeθ, strategy)
                         for (_loss, _set) in zip(datafree_bc_loss_function, bcs_train_sets)]
    else
        nothing
    end

    return pde_loss_functions, bc_loss_functions
end

function merge_strategy_with_loss_function(pinnrep::PINNRepresentation,
        strategy::GridTraining, datafree_pde_loss_function, datafree_bc_loss_function)
    (; domains, eqs, bcs, dict_indvars, dict_depvars) = pinnrep
    eltypeθ = recursive_eltype(pinnrep.flat_init_params)
    adaptor = EltypeAdaptor{eltypeθ}()

    train_sets = generate_training_sets(domains, strategy.dx, eqs, bcs, eltypeθ,
        dict_indvars, dict_depvars)

    # the points in the domain and on the boundary
    pde_train_sets, bcs_train_sets = train_sets
    pde_train_sets = adapt.(parameterless_type(ComponentArrays.getdata(flat_init_params)),
                            pde_train_sets)
    bcs_train_sets = adapt.(parameterless_type(ComponentArrays.getdata(flat_init_params)),
                            bcs_train_sets)

    pde_loss_functions = [get_loss_function(_loss, _set, eltypeθ, strategy)
                          for (_loss, _set) in zip(datafree_pde_loss_function,
                                                   pde_train_sets)]
    bc_loss_functions = [get_loss_function(_loss, _set, eltypeθ, strategy)
                         for (_loss, _set) in zip(datafree_bc_loss_function,
                                                   bcs_train_sets)]
    pde_loss_functions, bc_loss_functions
end

function get_loss_function(
        init_params, loss_function, train_set, eltype0, ::GridTraining; τ = nothing)
    init_params = init_params isa PINNRepresentation ? init_params.init_params : init_params
    train_set = train_set |> safe_get_device(init_params) |> EltypeAdaptor{eltype0}()
    return θ -> mean(abs2, loss_function(train_set, θ))
end

"""
    StochasticTraining(points; bcs_points = points)

## Positional Arguments

* `points`: number of points in random select training set

## Keyword Arguments

* `bcs_points`: number of points in random select training set for boundary conditions
  (by default, it equals `points`).
"""
struct StochasticTraining <: AbstractTrainingStrategy
    points::Int
    bcs_points::Int
end

StochasticTraining(points; bcs_points = points) = StochasticTraining(points, bcs_points)

function generate_random_points(points, bound, eltypeθ)
    lb, ub = bound
    return rand(eltypeθ, length(lb), points) .* (ub .- lb) .+ lb
end

function merge_strategy_with_loss_function(pinnrep::PINNRepresentation,
        strategy::StochasticTraining, datafree_pde_loss_function, datafree_bc_loss_function)
    (; domains, eqs, bcs, dict_indvars, dict_depvars) = pinnrep

    eltypeθ = eltype(pinnrep.flat_init_params)

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    pde_bounds, bcs_bounds = bounds

    pde_loss_functions = [get_loss_function(pinnrep, _loss, bound, eltypeθ, strategy)
                          for (_loss, bound) in zip(datafree_pde_loss_function, pde_bounds)]

    bc_loss_functions = [get_loss_function(pinnrep, _loss, bound, eltypeθ, strategy)
                         for (_loss, bound) in zip(datafree_bc_loss_function, bcs_bounds)]

    pde_loss_functions, bc_loss_functions
end

function get_loss_function(init_params, loss_function, bound, eltypeθ,
        strategy::StochasticTraining; τ = nothing)
    init_params = init_params isa PINNRepresentation ? init_params.init_params : init_params
    dev = safe_get_device(init_params)
    return θ -> begin
        sets = generate_random_points(strategy.points, bound, eltypeθ) |> dev |>
               EltypeAdaptor{recursive_eltype(θ)}()
        return mean(abs2, loss_function(sets, θ))
    end
end

"""
    QuasiRandomTraining(points; bcs_points = points,
                                sampling_alg = LatinHypercubeSample(), resampling = true,
                                minibatch = 0)


A training strategy which uses quasi-Monte Carlo sampling for low discrepancy sequences
that accelerate the convergence in high dimensional spaces over pure random sequences.

## Positional Arguments

* `points`:  the number of quasi-random points in a sample

## Keyword Arguments

* `bcs_points`: the number of quasi-random points in a sample for boundary conditions
  (by default, it equals `points`),
* `sampling_alg`: the quasi-Monte Carlo sampling algorithm,
* `resampling`: if it's false - the full training set is generated in advance before
  training, and at each iteration, one subset is randomly selected out of the batch.
  If it's true - the training set isn't generated beforehand, and one set of quasi-random
  points is generated directly at each iteration in runtime. In this case, `minibatch` has
  no effect.
* `minibatch`: the number of subsets, if `!resampling`.

For more information, see [QuasiMonteCarlo.jl](https://docs.sciml.ai/QuasiMonteCarlo/stable/).
"""
@concrete struct QuasiRandomTraining <: AbstractTrainingStrategy
    points::Int
    bcs_points::Int
    sampling_alg <: QuasiMonteCarlo.SamplingAlgorithm
    resampling::Bool
    minibatch::Int
end

function QuasiRandomTraining(points; bcs_points = points,
        sampling_alg = LatinHypercubeSample(), resampling = true, minibatch = 0)
    return QuasiRandomTraining(points, bcs_points, sampling_alg, resampling, minibatch)
end

function generate_quasi_random_points_batch(points, bound, eltypeθ, sampling_alg,
        minibatch)
    lb, ub = bound
    return QuasiMonteCarlo.generate_design_matrices(
        points, lb, ub, sampling_alg, minibatch) |> EltypeAdaptor{eltypeθ}()
end

function merge_strategy_with_loss_function(pinnrep::PINNRepresentation,
        strategy::QuasiRandomTraining, datafree_pde_loss_function,
        datafree_bc_loss_function)
    (; domains, eqs, bcs, dict_indvars, dict_depvars) = pinnrep

    eltypeθ = eltype(pinnrep.flat_init_params)

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    pde_bounds, bcs_bounds = bounds

    pde_loss_functions = [get_loss_function(pinnrep, _loss, bound, eltypeθ, strategy)
                          for (_loss, bound) in zip(datafree_pde_loss_function, pde_bounds)]

    strategy_ = QuasiRandomTraining(strategy.bcs_points; strategy.sampling_alg,
        strategy.resampling, strategy.minibatch)
    bc_loss_functions = [get_loss_function(pinnrep, _loss, bound, eltypeθ, strategy_)
                         for (_loss, bound) in zip(datafree_bc_loss_function, bcs_bounds)]

    return pde_loss_functions, bc_loss_functions
end

function get_loss_function(init_params, loss_function, bound, eltypeθ,
        strategy::QuasiRandomTraining; τ = nothing)
    (; sampling_alg, points, resampling, minibatch) = strategy

    init_params = init_params isa PINNRepresentation ? init_params.init_params : init_params
    dev = safe_get_device(init_params)

    return if resampling
        θ -> begin
            sets = @ignore_derivatives QuasiMonteCarlo.sample(
                points, bound[1], bound[2], sampling_alg)
            sets = sets |> dev |> EltypeAdaptor{eltypeθ}()
            return mean(abs2, loss_function(sets, θ))
        end
    else
        point_batch = generate_quasi_random_points_batch(
                          points, bound, eltypeθ, sampling_alg, minibatch) |> dev |>
                      EltypeAdaptor{eltypeθ}()
        θ -> mean(abs2, loss_function(point_batch[rand(1:minibatch)], θ))
    end
end

"""
    QuadratureTraining(; quadrature_alg = CubatureJLh(), reltol = 1e-6, abstol = 1e-3,
                        maxiters = 1_000, batch = 100)

A training strategy which treats the loss function as the integral of
||condition|| over the domain. Uses an Integrals.jl algorithm for
computing the (adaptive) quadrature of this loss with respect to the
chosen tolerances, with a batching `batch` corresponding to the maximum
number of points to evaluate in a given integrand call.

## Keyword Arguments

* `quadrature_alg`: quadrature algorithm,
* `reltol`: relative tolerance,
* `abstol`: absolute tolerance,
* `maxiters`: the maximum number of iterations in quadrature algorithm,
* `batch`: the preferred number of points to batch.

For more information on the argument values and algorithm choices, see
[Integrals.jl](https://docs.sciml.ai/Integrals/stable/).
"""
@concrete struct QuadratureTraining{T} <: AbstractTrainingStrategy
    quadrature_alg <: SciMLBase.AbstractIntegralAlgorithm
    reltol::T
    abstol::T
    maxiters::Int
    batch::Int
end

function QuadratureTraining(; quadrature_alg = CubatureJLh(), reltol = 1e-3, abstol = 1e-6,
        maxiters = 1_000, batch = 100)
    QuadratureTraining(quadrature_alg, reltol, abstol, maxiters, batch)
end

function merge_strategy_with_loss_function(pinnrep::PINNRepresentation,
        strategy::QuadratureTraining, datafree_pde_loss_function, datafree_bc_loss_function)
    (; domains, eqs, bcs, dict_indvars, dict_depvars) = pinnrep
    eltypeθ = eltype(pinnrep.flat_init_params)

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars, strategy)
    pde_bounds, bcs_bounds = bounds

    lbs, ubs = pde_bounds
    pde_loss_functions = [get_loss_function(pinnrep, _loss, lb, ub, eltypeθ, strategy)
                          for (_loss, lb, ub) in zip(datafree_pde_loss_function, lbs, ubs)]
    lbs, ubs = bcs_bounds
    bc_loss_functions = [get_loss_function(pinnrep, _loss, lb, ub, eltypeθ, strategy)
                         for (_loss, lb, ub) in zip(datafree_bc_loss_function, lbs, ubs)]

    return pde_loss_functions, bc_loss_functions
end

function get_loss_function(init_params, loss_function, lb, ub, eltypeθ,
        strategy::QuadratureTraining; τ = nothing)
    init_params = init_params isa PINNRepresentation ? init_params.init_params : init_params
    dev = safe_get_device(init_params)

    if length(lb) == 0
        return (θ) -> mean(abs2, loss_function(dev(rand(eltypeθ, 1, 10)), θ))
    end

    area = eltypeθ(prod(abs.(ub .- lb)))
    f_ = (lb, ub, loss_, θ) -> begin
        function integrand(x, θ)
            x = x |> dev |> EltypeAdaptor{eltypeθ}()
            return sum(abs2, view(loss_(x, θ), 1, :), dims = 2) #./ size_x
        end
        integral_function = BatchIntegralFunction(integrand, max_batch = strategy.batch)
        prob = IntegralProblem(integral_function, (lb, ub), θ)
        return solve(prob, strategy.quadrature_alg; strategy.reltol, strategy.abstol,
            strategy.maxiters)[1]
    end
    return (θ) -> f_(lb, ub, loss_function, θ) / area
end

"""
    WeightedIntervalTraining(weights, samples)

A training strategy that generates points for training based on the given inputs. 
We split the timespan into equal segments based on the number of weights, 
then sample points in each segment based on that segments corresponding weight,
such that the total number of sampled points is equivalent to the given samples

## Positional Arguments

* `weights`: A vector of weights that should sum to 1, representing the proportion of
  samples at each interval.
* `points`: the total number of samples that we want, across the entire time span

## Limitations

This training strategy can only be used with ODEs (`NNODE`).
"""
@concrete struct WeightedIntervalTraining{T} <: AbstractTrainingStrategy
    weights::Vector{T}
    points::Int
end

function get_loss_function(init_params, loss_function, train_set, eltype0,
        ::WeightedIntervalTraining; τ = nothing)
    init_params = init_params isa PINNRepresentation ? init_params.init_params : init_params
    train_set = train_set |> safe_get_device(init_params) |> EltypeAdaptor{eltype0}()
    return (θ) -> mean(abs2, loss_function(train_set, θ))
end
