"""
```julia
GridTraining(dx)
```

A training strategy that uses the grid points in a multidimensional grid
with spacings `dx`. If the grid is multidimensional, then `dx` is expected
to be an array of `dx` values matching the dimension of the domain,
corresponding to the grid spacing in each dimension.

## Positional Arguments

* `dx`: the discretization of the grid.
"""
struct GridTraining{T} <: AbstractTrainingStrategy
    dx::T
end

function merge_strategy_with_loss_function(pinnrep::PINNRepresentation,
                                           strategy::GridTraining,
                                           datafree_pde_loss_function,
                                           datafree_bc_loss_function)
    @unpack domains, eqs, bcs, varmap, flat_init_params = pinnrep
    dx = strategy.dx
    eltypeθ = eltype(pinnrep.flat_init_params)

    train_sets = generate_training_sets(domains, dx, eqs, bcs, eltypeθ,
                                        varmap)

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
                         for (_loss, _set) in zip(datafree_bc_loss_function, bcs_train_sets)]

    pde_loss_functions, bc_loss_functions
end

function get_loss_function(loss_function, train_set, eltypeθ, strategy::GridTraining;
                           τ = nothing)
    loss = (θ) -> mean(abs2, loss_function(train_set, θ))
end

"""
```julia
StochasticTraining(points; bcs_points = points)
```

## Positional Arguments

* `points`: number of points in random select training set

## Keyword Arguments

* `bcs_points`: number of points in random select training set for boundary conditions
  (by default, it equals `points`).
"""
struct StochasticTraining <: AbstractGridfreeStrategy
    points::Int64
    bcs_points::Int64
end

function StochasticTraining(points; bcs_points = points)
    StochasticTraining(points, bcs_points)
end

function generate_random_points(points, bound, eltypeθ)
    lb, ub = bound
    rand(eltypeθ, length(lb), points) .* (ub .- lb) .+ lb
end

function merge_strategy_with_loss_function(pinnrep::PINNRepresentation,
                                           strategy::StochasticTraining,
                                           datafree_pde_loss_function,
                                           datafree_bc_loss_function)
    @unpack domains, eqs, bcs, varmap, flat_init_params = pinnrep

    eltypeθ = eltype(pinnrep.flat_init_params)

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, varmap,
                        strategy)
    pde_bounds, bcs_bounds = bounds

    pde_loss_functions = [get_loss_function(_loss, bound, eltypeθ, strategy)
                          for (_loss, bound) in zip(datafree_pde_loss_function, pde_bounds)]

    bc_loss_functions = [get_loss_function(_loss, bound, eltypeθ, strategy)
                         for (_loss, bound) in zip(datafree_bc_loss_function, bcs_bounds)]

    pde_loss_functions, bc_loss_functions
end

function get_loss_function(loss_function, bound, eltypeθ, strategy::StochasticTraining;
                           τ = nothing)
    points = strategy.points

    loss = (θ) -> begin
        sets = generate_random_points(points, bound, eltypeθ)
        sets_ = adapt(parameterless_type(ComponentArrays.getdata(θ)), sets)
        mean(abs2, loss_function(sets_, θ))
    end
    return loss
end

"""
```julia
QuasiRandomTraining(points; bcs_points = points,
                            sampling_alg = LatinHypercubeSample(), resampling = true,
                            minibatch = 0)
```

A training strategy which uses quasi-Monte Carlo sampling for low discrepancy sequences
that accelerate the convergence in high dimensional spaces over pure random sequences.

## Positional Arguments

* `points`:  the number of quasi-random points in a sample

## Keyword Arguments

* `bcs_points`: the number of quasi-random points in a sample for boundary conditions
  (by default, it equals `points`),
* `sampling_alg`: the quasi-Monte Carlo sampling algorithm,
* `resampling`: if it's false - the full training set is generated in advance before training,
   and at each iteration, one subset is randomly selected out of the batch.
   Ff it's true - the training set isn't generated beforehand, and one set of quasi-random
   points is generated directly at each iteration in runtime. In this case, `minibatch` has no effect,
* `minibatch`: the number of subsets, if resampling == false.

For more information, see [QuasiMonteCarlo.jl](https://docs.sciml.ai/QuasiMonteCarlo/stable/)
"""
struct QuasiRandomTraining <: AbstractGridfreeStrategy
    points::Int64
    bcs_points::Int64
    sampling_alg::QuasiMonteCarlo.SamplingAlgorithm
    resampling::Bool
    minibatch::Int64
end

function QuasiRandomTraining(points; bcs_points = points,
                             sampling_alg = LatinHypercubeSample(), resampling = true,
                             minibatch = 0)
    QuasiRandomTraining(points, bcs_points, sampling_alg, resampling, minibatch)
end

function generate_quasi_random_points_batch(points, bound, eltypeθ, sampling_alg,
                                            minibatch)
    lb, ub = bound
    set = QuasiMonteCarlo.generate_design_matrices(points, lb, ub, sampling_alg, minibatch)
    set = map(s -> adapt(parameterless_type(eltypeθ), s), set)
    return set
end

function merge_strategy_with_loss_function(pinnrep::PINNRepresentation,
                                           strategy::QuasiRandomTraining,
                                           datafree_pde_loss_function,
                                           datafree_bc_loss_function)
    @unpack domains, eqs, bcs, varmap, flat_init_params = pinnrep

    eltypeθ = eltype(pinnrep.flat_init_params)

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, varmap,
                        strategy)
    pde_bounds, bcs_bounds = bounds

    pde_loss_functions = [get_loss_function(_loss, bound, eltypeθ, strategy)
                          for (_loss, bound) in zip(datafree_pde_loss_function, pde_bounds)]

    strategy_ = QuasiRandomTraining(strategy.bcs_points;
                                    sampling_alg = strategy.sampling_alg,
                                    resampling = strategy.resampling,
                                    minibatch = strategy.minibatch)
    bc_loss_functions = [get_loss_function(_loss, bound, eltypeθ, strategy_)
                         for (_loss, bound) in zip(datafree_bc_loss_function, bcs_bounds)]

    pde_loss_functions, bc_loss_functions
end

function get_loss_function(loss_function, bound, eltypeθ, strategy::QuasiRandomTraining;
                           τ = nothing)
    sampling_alg = strategy.sampling_alg
    points = strategy.points
    resampling = strategy.resampling
    minibatch = strategy.minibatch

    point_batch = nothing
    point_batch = if resampling == false
        generate_quasi_random_points_batch(points, bound, eltypeθ, sampling_alg, minibatch)
    end
    loss = if resampling == true
        θ -> begin
            sets = ChainRulesCore.@ignore_derivatives QuasiMonteCarlo.sample(points,
                                                                             bound[1],
                                                                             bound[2],
                                                                             sampling_alg)
            sets_ = adapt(parameterless_type(ComponentArrays.getdata(θ)), sets)
            mean(abs2, loss_function(sets_, θ))
        end
    else
        θ -> begin
            sets_ = point_batch[rand(1:minibatch)]
            sets__ = adapt(parameterless_type(ComponentArrays.getdata(θ)), sets_)
            mean(abs2, loss_function(sets__, θ))
        end
    end
    return loss
end

"""
```julia
QuadratureTraining(; quadrature_alg = CubatureJLh(),
                     reltol = 1e-6, abstol = 1e-3,
                     maxiters = 1_000, batch = 100)
```

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
struct QuadratureTraining{Q <: SciMLBase.AbstractIntegralAlgorithm, T} <:
       AbstractGridfreeStrategy
    quadrature_alg::Q
    reltol::T
    abstol::T
    maxiters::Int64
    batch::Int64
end

function QuadratureTraining(; quadrature_alg = CubatureJLh(), reltol = 1e-6, abstol = 1e-3,
                            maxiters = 1_000, batch = 100)
    QuadratureTraining(quadrature_alg, reltol, abstol, maxiters, batch)
end

function merge_strategy_with_loss_function(pinnrep::PINNRepresentation,
                                           strategy::QuadratureTraining,
                                           datafree_pde_loss_function,
                                           datafree_bc_loss_function)
    @unpack domains, eqs, bcs, varmap, flat_init_params = pinnrep
    eltypeθ = eltype(pinnrep.flat_init_params)

    bounds = get_bounds(domains, eqs, bcs, eltypeθ, varmap,
                        strategy)
    pde_bounds, bcs_bounds = bounds

    lbs, ubs = pde_bounds
    pde_loss_functions = [get_loss_function(_loss, lb, ub, eltypeθ, strategy)
                          for (_loss, lb, ub) in zip(datafree_pde_loss_function, lbs, ubs)]
    lbs, ubs = bcs_bounds
    bc_loss_functions = [get_loss_function(_loss, lb, ub, eltypeθ, strategy)
                         for (_loss, lb, ub) in zip(datafree_bc_loss_function, lbs, ubs)]

    pde_loss_functions, bc_loss_functions
end

function get_loss_function(loss_function, lb, ub, eltypeθ, strategy::QuadratureTraining;
                           τ = nothing)
    if length(lb) == 0
        loss = (θ) -> mean(abs2, loss_function(rand(eltypeθ, 1, 10), θ))
        return loss
    end
    area = eltypeθ(prod(abs.(ub .- lb)))
    f_ = (lb, ub, loss_, θ) -> begin
        # last_x = 1
        function integrand(x, θ)
            # last_x = x
            # mean(abs2,loss_(x,θ), dims=2)
            # size_x = fill(size(x)[2],(1,1))
            x = adapt(parameterless_type(ComponentArrays.getdata(θ)), x)
            sum(abs2, loss_(x, θ), dims = 2) #./ size_x
        end
        prob = IntegralProblem(integrand, lb, ub, θ, batch = strategy.batch, nout = 1)
        solve(prob,
              strategy.quadrature_alg,
              reltol = strategy.reltol,
              abstol = strategy.abstol,
              maxiters = strategy.maxiters)[1]
    end
    loss = (θ) -> 1 / area * f_(lb, ub, loss_function, θ)
    return loss
end

"""
```julia
WeightedIntervalTraining(weights, samples)
```

A training strategy that generates points for training based on the given inputs.
We split the timespan into equal segments based on the number of weights,
then sample points in each segment based on that segments corresponding weight,
such that the total number of sampled points is equivalent to the given samples

## Positional Arguments

* `weights`: A vector of weights that should sum to 1, representing the proportion of samples at each interval.
* `samples`: the total number of samples that we want, across the entire time span

## Limitations

This training strategy can only be used with ODEs (`NNODE`).
"""
struct WeightedIntervalTraining{T} <: AbstractGridfreeStrategy
    weights::Vector{T}
    samples::Int
end

function WeightedIntervalTraining(weights, samples)
    WeightedIntervalTraining(weights, samples)
end

function get_loss_function(loss_function, train_set, eltypeθ,
                           strategy::WeightedIntervalTraining;
                           τ = nothing)
    loss = (θ) -> mean(abs2, loss_function(train_set, θ))
end
