"""
    AbstractDerivativeLowering

Abstract supertype for the strategies used to lower `Differential` operators applied to
neural-network trial functions into numeric expressions.

# Extension Rules

A concrete subtype must add a method for
`lower_derivative(::MyLowering, inner, ctx, shift, direction, order)` returning the
symbolic array expression for the `order`-th derivative of the lowered `inner` expression
along collocation row `direction`.
"""
abstract type AbstractDerivativeLowering end

"""
    FiniteDifferenceDerivative()

Lower derivatives of trial functions with central finite-difference stencils applied to the
neural network inputs. Pure derivatives of order one through four use the standard
central stencils; higher orders and mixed derivatives are built recursively. The step
size for an `order`-th derivative is `eps(T)^(1 / (2 + order))` where `T` is the element
type of the network parameters.
"""
struct FiniteDifferenceDerivative <: AbstractDerivativeLowering end

"""
    nn_eval(NN, X, θ)

Evaluate the symbolic neural network `NN` on the batch of inputs `X` (one column per
point) with the flat parameter vector `θ`. Symbolically the result is a
`size(NN, 1) × size(X, 2)` array whose representation size does not depend on the number
of collocation points; numerically it calls the network's `stateless_apply` wrapper.
"""
function nn_eval end
Symbolics.@register_array_symbolic nn_eval(f::Any, X::AbstractMatrix, θ::AbstractVector) begin
    size = (size(f, 1), size(X, 2))
    eltype = Real
end false
SymbolicUtils.promote_symtype(::typeof(nn_eval), f, X, θ) = Array{Real, 2}
function SymbolicUtils.promote_shape(
        ::typeof(nn_eval), shf::SymbolicUtils.ShapeT, shX::SymbolicUtils.ShapeT,
        shθ::SymbolicUtils.ShapeT
    )
    if shf isa SymbolicUtils.Unknown || shX isa SymbolicUtils.Unknown
        return SymbolicUtils.Unknown(2)
    end
    return SymbolicUtils.ShapeVecT([1:length(shf[1]), 1:length(shX[2])])
end
nn_eval(f, X, θ) = f(X, θ)

"""
    nn_eval_row(NN, X, θ, k)

The `k`-th output row of [`nn_eval`](@ref) as a `1 × size(X, 2)` array. Used for
dependent variables that share one multi-output network.
"""
function nn_eval_row end
Symbolics.@register_array_symbolic nn_eval_row(
    f::Any, X::AbstractMatrix, θ::AbstractVector, k::Integer
) begin
    size = (1, size(X, 2))
    eltype = Real
end false
SymbolicUtils.promote_symtype(::typeof(nn_eval_row), f, X, θ, k) = Array{Real, 2}
function SymbolicUtils.promote_shape(
        ::typeof(nn_eval_row), shf::SymbolicUtils.ShapeT, shX::SymbolicUtils.ShapeT,
        shθ::SymbolicUtils.ShapeT, shk::SymbolicUtils.ShapeT
    )
    shX isa SymbolicUtils.Unknown && return SymbolicUtils.Unknown(2)
    return SymbolicUtils.ShapeVecT([1:1, 1:length(shX[2])])
end
nn_eval_row(f, X, θ, k) = f(X, θ)[k:k, :]

"""
    default_adtype()

The automatic differentiation backend `discretize` uses unless `adtype` is given:
`AutoZygote()`. Enzyme can be selected with
`adtype = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation = Enzyme.Const)`;
static activity analysis rejects the generated objective, and with runtime activity the
reverse pass through an `additional_loss` closure was observed to overwrite arrays the
closure captures, so Enzyme is not the default until that is resolved. The
Reactant-compiled path (see the PDE tutorial) differentiates the generated objective with
Enzyme correctly.
"""
default_adtype() = AutoZygote()

"""
    PhysicsInformedNN(chain, strategy; kwargs...)

A discretization of a `PDESystem` that represents each dependent variable with a neural
network and lowers the PDE residuals and boundary conditions into an optimization
`System` whose unknowns are the network parameters.

`symbolic_discretize(pdesys, discretization)` returns the `ModelingToolkit.System`;
`discretize(pdesys, discretization)` compiles it into an `OptimizationProblem`.

## Positional Arguments

* `chain`: a Lux layer, or a vector with one Lux layer per dependent variable of the
  `PDESystem`. A single layer with several dependent variables is treated as one shared
  network whose `i`-th output is the `i`-th dependent variable.
* `strategy`: the [`AbstractTrainingStrategy`](@ref) that chooses the collocation points
  and the reduction of the pointwise residuals.

## Keyword Arguments

* `init_params`: initial network parameters as a flat vector, a Lux parameter
  `NamedTuple` or a `ComponentArray` (one network), or a vector of those (one per
  network). Defaults to `Lux.initialparameters` with `rng`.
* `rng`: the random number generator used for parameter initialization and sampling.
* `derivative`: the [`AbstractDerivativeLowering`](@ref) used for `Differential`
  operators. Defaults to [`FiniteDifferenceDerivative`](@ref).
* `param_estim`: when `true`, the parameters of the `PDESystem` are added to the
  unknowns of the generated `System` and co-optimized with the network weights.
* `additional_loss`: an optional function `(phi, θ, p) -> loss` added as an extra cost.
  `phi` is a `NamedTuple` of batched network evaluators keyed by dependent variable name,
  `θ` a `NamedTuple` of parameter vectors with the same keys and `p` the vector of
  `PDESystem` parameter values.
* `boundary_policy`: how boundary conditions enter the `System`. `:penalty` (default)
  adds the mean squared boundary residual as a cost; `:constraints` keeps the pointwise
  boundary residuals as equality constraints of the `System`.
* `eval_points`: number of points per independent variable in the evaluation grid used by
  the solution interface (`sol[u(x, t)]`).

## Example

```julia
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)
bcs = [u(0, y) ~ 0, u(1, y) ~ 0, u(x, 0) ~ 0, u(x, 1) ~ 0]
domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
@named pdesys = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
chain = Chain(Dense(2, 16, σ), Dense(16, 16, σ), Dense(16, 1))
disc = PhysicsInformedNN(chain, GridTraining(0.05))
prob = discretize(pdesys, disc)
sol = solve(prob, Adam(0.01); maxiters = 1000)
```
"""
@concrete struct PhysicsInformedNN <: PDEBase.AbstractOptimizationSystemDiscretization
    chain
    strategy <: AbstractTrainingStrategy
    init_params
    rng <: AbstractRNG
    derivative <: AbstractDerivativeLowering
    param_estim::Bool
    additional_loss
    boundary_policy::Symbol
    eval_points::Int
end

function PhysicsInformedNN(
        chain, strategy::AbstractTrainingStrategy; init_params = nothing,
        rng::AbstractRNG = Random.default_rng(), derivative = FiniteDifferenceDerivative(),
        param_estim::Bool = false, additional_loss = nothing,
        boundary_policy::Symbol = :penalty, eval_points::Int = 100
    )
    boundary_policy in (:penalty, :constraints) || throw(
        ArgumentError(
            "`boundary_policy` must be `:penalty` or `:constraints`, got `$(boundary_policy)`."
        )
    )
    chain = chain isa AbstractArray ? map(_to_lux, chain) : _to_lux(chain)
    return PhysicsInformedNN(
        chain, strategy, init_params, rng, derivative, param_estim, additional_loss,
        boundary_policy, eval_points
    )
end

_to_lux(layer::AbstractLuxLayer) = layer
function _to_lux(layer)
    Base.depwarn(
        "Passing non-Lux layers to `PhysicsInformedNN` is deprecated; convert with \
        `Lux.FromFluxAdaptor` or use a Lux layer directly.", :PhysicsInformedNN
    )
    return FromFluxAdaptor()(layer)
end

PDEBase.get_time(::PhysicsInformedNN) = nothing

"""
    TrialNetwork

The symbolic neural network representing one dependent variable.

# Fields

* `depvar`: the dependent variable operation (e.g. `u` for `u(x, t)`).
* `args`: the independent variables the dependent variable is called with.
* `NN`: the symbolic network callable parameter (see `ModelingToolkitNeuralNets`).
* `θ`: the array unknown holding the flat network parameters.
* `output`: the output row of `NN` used for this dependent variable.
* `noutputs`: the number of outputs of `NN`.
* `chain`: the Lux layer.
"""
struct TrialNetwork{D, A, N, P, C}
    depvar::D
    args::A
    NN::N
    θ::P
    output::Int
    noutputs::Int
    chain::C
end

"""
    ResidualBlock

One equation of the `PDESystem` (a PDE or a boundary condition) lowered onto its own set
of collocation points.

# Fields

* `eq`: the original equation.
* `kind`: `:pde` or `:bc`.
* `ivs`: the free independent variables sampled for this equation, in collocation row
  order.
* `ivpos`: the position of each free independent variable in `get_ivs(pdesys)`.
* `bounds`: `(lb, ub)` for the free independent variables.
* `pinned`: for each free independent variable, the set of values it is pinned to by
  boundary conditions (used by `GridTraining` to exclude them from PDE interiors).
* `xs`: the array parameter holding the collocation points, or `nothing` when the
  equation has no free independent variable.
* `w`: the array parameter holding quadrature weights, or `nothing` for strategies that
  use the mean squared residual.
* `npoints`: number of collocation points.
* `residual`: the lowered symbolic residual (a `1 × npoints` array expression).
"""
struct ResidualBlock{E, I, B, X, W, R}
    eq::E
    kind::Symbol
    ivs::I
    ivpos::Vector{Int}
    bounds::B
    pinned::Vector{Set{Float64}}
    xs::X
    w::W
    npoints::Int
    residual::R
end

"""
    PINNMetadata

Discretization metadata attached to the `System` produced by
`symbolic_discretize(pdesys, ::PhysicsInformedNN)`. `SciMLBase.wrap_sol` uses it to turn
the `OptimizationSolution` into a `PDENoTimeSolution`.

# Fields

* `pdesys`: the discretized `PDESystem`.
* `disc`: the [`PhysicsInformedNN`](@ref) discretization.
* `varmap`: the `PDEBase.VariableMap` of the system.
* `networks`: the [`TrialNetwork`](@ref) of each dependent variable.
* `blocks`: the [`ResidualBlock`](@ref) of each PDE and boundary condition.
* `ps`: the symbolic `PDESystem` parameters in the order of `pdesys.ps`.
* `eval_grid`: `Dict` from independent variable to its evaluation grid.
* `metadata`: a `Ref` holding the compiled `System`, filled by `discretize`.
"""
struct PINNMetadata{P, D, V, N, S, G} <:
    SciMLBase.AbstractDiscretizationMetadata{Val{false}()}
    pdesys::P
    disc::D
    varmap::V
    networks::N
    blocks::Vector{ResidualBlock}
    ps::S
    eval_grid::G
    metadata::Ref{Any}
end

function PINNMetadata(pdesys, disc, varmap, networks, blocks, ps, eval_grid)
    return PINNMetadata(pdesys, disc, varmap, networks, blocks, ps, eval_grid, Ref{Any}(nothing))
end

PDEBase.add_metadata!(md::PINNMetadata, sys) = (md.metadata[] = sys)

"""
    AdditionalLoss

Callable parameter wrapping a user `additional_loss(phi, θ, p)` so that it can appear as
a cost of the generated `System`. It is called with the network parameter vectors
followed by the `PDESystem` parameter values.
"""
struct AdditionalLoss{F, K, N, O}
    f::F
    names::K
    wrappers::N
    outputs::O
end

function (al::AdditionalLoss)(args...)
    nnets = length(al.names)
    θs = NamedTuple{al.names}(ntuple(i -> args[i], nnets))
    phi = NamedTuple{al.names}(
        ntuple(nnets) do i
            w = al.wrappers[i]
            k = al.outputs[i]
            (X, θ) -> w(X, θ)[k:k, :]
        end
    )
    p = collect(args[(nnets + 1):end])
    return al.f(phi, θs, p)
end
