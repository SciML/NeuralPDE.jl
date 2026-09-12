"""
    GridTraining(dx)

A training strategy that uses the points of a multidimensional grid with spacings `dx`.
If the grid is multidimensional, `dx` may be an array of spacings matching the
independent variables of the `PDESystem` in order.

For PDEs the interior grid excludes the coordinate values pinned by the boundary
conditions, and the cost of each equation is the mean of its squared residuals.

## Positional Arguments

* `dx`: the discretization of the grid.
"""
@concrete struct GridTraining <: AbstractTrainingStrategy
    dx
end

_zero_dimensional_coordinates(::Type{T}) where {T} = Matrix{T}(undef, 0, 1)

"""
    get_loss_function(init_params, loss_function, training_data, eltype, strategy; kwargs...)
    get_loss_function(
        init_params, loss_function, lower_bounds, upper_bounds, eltype, strategy;
        kwargs...
    )

Construct the scalar objective used by the ODE solvers (`NNODE`, `NNDAE`, `NNSDE`) from a
residual function and strategy-specific training data.

# Arguments

* `init_params`: the initial parameter container. The strategy uses this to choose a
  device when needed.
* `loss_function`: a function called as `loss_function(training_data, θ)` that returns
  the residuals for the current optimization parameters `θ`.
* `training_data`: the grid, sampled points, or bounds consumed by the strategy.
* `lower_bounds`, `upper_bounds`: the lower and upper integration bounds for strategies
  that use the interval form of the interface.
* `eltype`: the element type used for generated training data.
* `strategy`: an `AbstractTrainingStrategy` selecting the extension method.

# Keyword Arguments

* `kwargs`: strategy-specific options. They are forwarded to the selected extension
  method.

# Returns

A callable `objective(θ)` that returns the scalar training objective for `θ`.

# Extension Rules

Custom strategies extend this generic function with a method specialized on their
strategy type. The method must return a callable whose first argument is the
optimization parameter container.

# Examples

```julia
struct MyTraining <: AbstractTrainingStrategy end

function get_loss_function(
        init_params, loss_function, training_data, T, ::MyTraining; scale = one(T)
    )
    return θ -> scale * sum(abs2, loss_function(training_data, θ))
end
```
"""
function get_loss_function end

function get_loss_function(
        init_params, loss_function, train_set, eltype0, ::GridTraining; τ = nothing
    )
    train_set = train_set |> safe_get_device(init_params) |> EltypeAdaptor{eltype0}()
    return θ -> mean(abs2, loss_function(train_set, θ))
end

"""
    StochasticTraining(points; bcs_points = points)

A training strategy that draws `points` uniformly distributed random points in the
domain of each equation. For PDEs the points are stored as parameters of the generated
`OptimizationProblem`; call [`resample!`](@ref) from a `solve` callback to draw new points
during training.

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
    isempty(lb) && return _zero_dimensional_coordinates(eltypeθ)
    return rand(eltypeθ, length(lb), points) .* (ub .- lb) .+ lb
end

function get_loss_function(
        init_params, loss_function, bound, eltypeθ,
        strategy::StochasticTraining; τ = nothing
    )
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
* `resampling`: whether [`resample!`](@ref) draws a new sample for PDE problems. For the
  ODE solvers, `false` generates `minibatch` samples in advance and selects one of them at
  random on every objective evaluation.
* `minibatch`: the number of subsets, if `!resampling` (ODE solvers only).

For more information, see [QuasiMonteCarlo.jl](https://docs.sciml.ai/QuasiMonteCarlo/stable/).
"""
@concrete struct QuasiRandomTraining <: AbstractTrainingStrategy
    points::Int
    bcs_points::Int
    sampling_alg <: QuasiMonteCarlo.SamplingAlgorithm
    resampling::Bool
    minibatch::Int
end

function QuasiRandomTraining(
        points; bcs_points = points,
        sampling_alg = LatinHypercubeSample(), resampling = true, minibatch = 0
    )
    return QuasiRandomTraining(points, bcs_points, sampling_alg, resampling, minibatch)
end

function generate_quasi_random_points_batch(
        points, bound, eltypeθ, sampling_alg, minibatch
    )
    lb, ub = bound
    set = QuasiMonteCarlo.generate_design_matrices(points, lb, ub, sampling_alg, minibatch)
    return set |> EltypeAdaptor{eltypeθ}()
end

function get_loss_function(
        init_params, loss_function, bound, eltypeθ,
        strategy::QuasiRandomTraining; τ = nothing
    )
    (; sampling_alg, points, resampling, minibatch) = strategy
    dev = safe_get_device(init_params)

    if isempty(bound[1])
        coordinates = dev(_zero_dimensional_coordinates(eltypeθ))
        return θ -> mean(abs2, loss_function(coordinates, θ))
    end

    return if resampling
        θ -> begin
            sets = @ignore_derivatives QuasiMonteCarlo.sample(
                points, bound[1], bound[2], sampling_alg
            )
            sets = sets |> dev |> EltypeAdaptor{eltypeθ}()
            return mean(abs2, loss_function(sets, θ))
        end
    else
        point_batch = generate_quasi_random_points_batch(
            points, bound, eltypeθ, sampling_alg, minibatch
        ) |> dev |>
            EltypeAdaptor{eltypeθ}()
        θ -> mean(abs2, loss_function(point_batch[rand(1:minibatch)], θ))
    end
end

"""
    QuadratureTraining(; quadrature_alg = CubatureJLh(), reltol = 1e-6, abstol = 1e-3,
                        maxiters = 1_000, batch = 100)

A training strategy which treats the loss function as the integral of
||condition|| over the domain.

For the ODE solvers, the integral is computed with the adaptive Integrals.jl algorithm
`quadrature_alg` under the given tolerances, batching at most `batch` points per
integrand call.

For PDEs, `quadrature_alg` must be a fixed-node `Integrals.GaussLegendre` rule; the
tensor-product Gauss–Legendre nodes are the collocation points of every equation and the
cost is the quadrature of the squared residual normalized by the domain measure. The
tolerance keywords are ignored.

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

function QuadratureTraining(;
        quadrature_alg = CubatureJLh(), reltol = 1.0e-3, abstol = 1.0e-6,
        maxiters = 1_000, batch = 100
    )
    return QuadratureTraining(quadrature_alg, reltol, abstol, maxiters, batch)
end

function get_loss_function(
        init_params, loss_function, lb, ub, eltypeθ,
        strategy::QuadratureTraining; τ = nothing
    )
    dev = safe_get_device(init_params)

    if length(lb) == 0
        # Fixed numeric arguments use the first row to inherit the batch shape.
        coordinates = dev(zeros(eltypeθ, 1, 1))
        return θ -> mean(abs2, loss_function(coordinates, θ))
    end

    area = eltypeθ(prod(abs.(ub .- lb)))
    f_ = (
        lb,
        ub,
        loss_,
        θ,
    ) -> begin
        function integrand(x, θ)
            x = x |> dev |> EltypeAdaptor{eltypeθ}()
            isempty(x) && return similar(x, recursive_eltype(θ), 0)
            return sum(abs2, view(loss_(x, θ), 1, :), dims = 2) #./ size_x
        end
        integral_function = BatchIntegralFunction(integrand, max_batch = strategy.batch)
        prob = IntegralProblem(integral_function, (lb, ub), θ)
        return solve(
            prob, strategy.quadrature_alg; strategy.reltol, strategy.abstol,
            strategy.maxiters
        ).u
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

function get_loss_function(
        init_params, loss_function, train_set, eltype0,
        ::WeightedIntervalTraining; τ = nothing
    )
    train_set = train_set |> safe_get_device(init_params) |> EltypeAdaptor{eltype0}()
    return θ -> mean(abs2, loss_function(train_set, θ))
end

# PDE collocation interface. Every strategy answers how many points a residual block
# gets, how to draw them, and whether `resample!` should redraw them.

"""
    collocation_count(strategy, kind, ivpos, bounds, pinned)

Number of collocation points a residual block of kind `:pde` or `:bc` receives under
`strategy`. `ivpos` holds the positions of the block's free independent variables in
`get_ivs(pdesys)`, `bounds` is `(lb, ub)` and `pinned` the set of boundary-pinned values
of each free independent variable.
"""
function collocation_count end

"""
    sample_points(strategy, block::ResidualBlock, rng)

Return `(X, W)`: the `d × n` matrix of collocation points and the `1 × n` quadrature
weights (or `nothing`) of `block` under `strategy`.
"""
function sample_points end

"""
    resamples(strategy)

Whether [`resample!`](@ref) draws new collocation points for `strategy`.
"""
resamples(::AbstractTrainingStrategy) = false

"""
    uses_quadrature_weights(strategy)

Whether residual blocks of `strategy` carry a quadrature weight parameter.
"""
uses_quadrature_weights(::AbstractTrainingStrategy) = false

_pde_strategy_error(strategy) = throw(
    ArgumentError(
        "`$(typeof(strategy).name.name)` cannot be used with `PhysicsInformedNN`; use \
        `GridTraining`, `StochasticTraining`, `QuasiRandomTraining` or `QuadratureTraining`."
    )
)
collocation_count(strategy::AbstractTrainingStrategy, args...) = _pde_strategy_error(strategy)
sample_points(strategy::AbstractTrainingStrategy, args...) = _pde_strategy_error(strategy)

function grid_axes(strategy::GridTraining, kind, ivpos, bounds, pinned)
    lb, ub = bounds
    T = eltype(lb)
    return map(eachindex(ivpos)) do i
        dx = strategy.dx isa Number ? strategy.dx : strategy.dx[ivpos[i]]
        axis = collect(T, lb[i]:T(dx):ub[i])
        kind == :pde && filter!(v -> !(v in pinned[i]), axis)
        axis
    end
end

function collocation_count(strategy::GridTraining, kind, ivpos, bounds, pinned)
    isempty(ivpos) && return 1
    return prod(length, grid_axes(strategy, kind, ivpos, bounds, pinned))
end

function sample_points(strategy::GridTraining, block::ResidualBlock, rng)
    axes_ = grid_axes(strategy, block.kind, block.ivpos, block.bounds, block.pinned)
    return _product_matrix(axes_), nothing
end

function _product_matrix(axes_)
    d = length(axes_)
    n = prod(length, axes_)
    X = Matrix{eltype(first(axes_))}(undef, d, n)
    for (j, pt) in enumerate(Iterators.product(axes_...))
        X[:, j] .= pt
    end
    return X
end

function collocation_count(strategy::StochasticTraining, kind, ivpos, bounds, pinned)
    isempty(ivpos) && return 1
    return kind == :pde ? strategy.points : strategy.bcs_points
end

function sample_points(::StochasticTraining, block::ResidualBlock, rng)
    lb, ub = block.bounds
    X = rand(rng, eltype(lb), length(lb), block.npoints) .* (ub .- lb) .+ lb
    return X, nothing
end

resamples(::StochasticTraining) = true

function collocation_count(strategy::QuasiRandomTraining, kind, ivpos, bounds, pinned)
    isempty(ivpos) && return 1
    return kind == :pde ? strategy.points : strategy.bcs_points
end

function sample_points(strategy::QuasiRandomTraining, block::ResidualBlock, rng)
    lb, ub = block.bounds
    X = QuasiMonteCarlo.sample(block.npoints, lb, ub, strategy.sampling_alg)
    return Matrix{eltype(lb)}(reshape(X, length(lb), block.npoints)), nothing
end

resamples(strategy::QuasiRandomTraining) = strategy.resampling

function _gauss_legendre_order(strategy::QuadratureTraining)
    alg = strategy.quadrature_alg
    alg isa GaussLegendre || throw(
        ArgumentError(
            "`QuadratureTraining` for PDEs requires a fixed-node rule; pass \
            `quadrature_alg = GaussLegendre(n = ...)`."
        )
    )
    return length(alg.nodes)
end

function collocation_count(strategy::QuadratureTraining, kind, ivpos, bounds, pinned)
    isempty(ivpos) && return 1
    return _gauss_legendre_order(strategy)^length(ivpos)
end

uses_quadrature_weights(::QuadratureTraining) = true

function sample_points(strategy::QuadratureTraining, block::ResidualBlock, rng)
    lb, ub = block.bounds
    T = eltype(lb)
    nodes, weights = gausslegendre(_gauss_legendre_order(strategy))
    axes_ = [T.((ub[i] - lb[i]) / 2 .* nodes .+ (ub[i] + lb[i]) / 2) for i in eachindex(lb)]
    waxes = [T.(weights ./ 2) for _ in eachindex(lb)]
    X = _product_matrix(axes_)
    W = reshape(vec(_product_matrix(waxes) |> w -> prod(w; dims = 1)), 1, :)
    return X, W
end
