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
    EnzymeForwardDerivative()

Lower spatial derivatives of batched neural-network trial functions with nested
Enzyme forward-mode Jacobian-vector products. Supports pure and mixed derivatives
of total order one through four, including derivatives at literal boundary points.
Network arguments must be independent variables or numeric literals. Networks must
act independently on each batch column and support Enzyme differentiation.

Use `PhysicsInformedNN(chain, strategy; derivative = EnzymeForwardDerivative())`.
The derivative backend is independent of the outer optimization `adtype`; Enzyme
reverse mode and Zygote (through a ChainRules pullback) differentiate its output.
"""
struct EnzymeForwardDerivative <: AbstractDerivativeLowering end

"""
    nn_jvp(NN, X, θ, directions, k)

Evaluate row `k` of nested spatial Jacobian-vector products of the batched network
`NN(X, θ)`. `directions` is a `Val` containing a tuple of tuples of input row indices;
each inner tuple seeds those rows with ones across the batch. The result is a
`1 × size(X, 2)` array, represented symbolically by one registered array call.
Network parameters reside in `θ`; `NN` and `directions` are constant configuration.
"""
function nn_jvp end
Symbolics.@register_array_symbolic nn_jvp(
    f::Any, X::AbstractMatrix, θ::AbstractVector, directions::Any, k::Integer
) begin
    size = (1, size(X, 2))
    eltype = Real
end false
SymbolicUtils.promote_symtype(::typeof(nn_jvp), f, X, θ, directions, k) = Array{Real, 2}
function SymbolicUtils.promote_shape(
        ::typeof(nn_jvp), shf::SymbolicUtils.ShapeT, shX::SymbolicUtils.ShapeT,
        shθ::SymbolicUtils.ShapeT, shdirections::SymbolicUtils.ShapeT,
        shk::SymbolicUtils.ShapeT
    )
    shX isa SymbolicUtils.Unknown && return SymbolicUtils.Unknown(2)
    return SymbolicUtils.ShapeVecT([1:1, 1:length(shX[2])])
end

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

# Registration keeps symbolic simplification from materializing a host constant array.
function _constant_row end
Symbolics.@register_array_symbolic _constant_row(X::AbstractArray, value::Number) begin
    size = (1, ndims(X) == 1 ? 1 : size(X, 2))
    eltype = Real
end false
SymbolicUtils.promote_symtype(::typeof(_constant_row), X, value) = Array{Real, 2}
function SymbolicUtils.promote_shape(
        ::typeof(_constant_row), shX::SymbolicUtils.ShapeT, shvalue::SymbolicUtils.ShapeT
    )
    shX isa SymbolicUtils.Unknown && return SymbolicUtils.Unknown(2)
    return SymbolicUtils.ShapeVecT([1:1, 1:(length(shX) == 1 ? 1 : length(shX[2]))])
end
_constant_row(X, value) = zero.(X[1:1, :]) .+ value

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
    nn_vcat(A, B)

Lazy `vcat` for symbolic matrix expressions: row-stacking of `1 × n` lowered
expressions into a `k × n` input matrix. `Base.vcat` materializes every element when
applied to symbolic arrays, so dependent-variable calls with general argument
expressions stack their argument rows through this registered call instead, keeping
the representation size independent of the batch size.
"""
function nn_vcat end
Symbolics.@register_array_symbolic nn_vcat(A::AbstractMatrix, B::AbstractMatrix) begin
    size = (size(A, 1) + size(B, 1), size(A, 2))
    eltype = Real
end false
SymbolicUtils.promote_symtype(::typeof(nn_vcat), A, B) = Array{Real, 2}
function SymbolicUtils.promote_shape(
        ::typeof(nn_vcat), shA::SymbolicUtils.ShapeT, shB::SymbolicUtils.ShapeT
    )
    if shA isa SymbolicUtils.Unknown || shB isa SymbolicUtils.Unknown
        return SymbolicUtils.Unknown(2)
    end
    m = length(shA[1]) + length(shB[1])
    return SymbolicUtils.ShapeVecT([1:m, 1:length(shA[2])])
end
nn_vcat(A, B) = vcat(A, B)

"""
    nn_veccat(a, b)

Lazy `vcat` for symbolic vectors; the vector counterpart of [`nn_vcat`](@ref), used to
concatenate network parameter vectors and scalar `PDESystem` parameters into the flat
argument of [`quadrature`](@ref).
"""
function nn_veccat end
Symbolics.@register_array_symbolic nn_veccat(A::AbstractVector, B::AbstractVector) begin
    size = (length(A) + length(B),)
    eltype = Real
end false
SymbolicUtils.promote_symtype(::typeof(nn_veccat), A, B) = Vector{Real}
function SymbolicUtils.promote_shape(
        ::typeof(nn_veccat), shA::SymbolicUtils.ShapeT, shB::SymbolicUtils.ShapeT
    )
    if shA isa SymbolicUtils.Unknown || shB isa SymbolicUtils.Unknown
        return SymbolicUtils.Unknown(1)
    end
    return SymbolicUtils.ShapeVecT([1:(length(shA[1]) + length(shB[1]))])
end
nn_veccat(A, B) = vcat(A, B)

"""
    quadrature(f, X, θ, ξ, w)

Fixed-node quadrature of a batched integrand over the collocation points. `X` is the
`d × n` matrix of collocation points, `θ` the flat network-parameter vector, `ξ` the
`q × M` matrix of tensor-product quadrature nodes on the reference hypercube `[-1, 1]^q`
and `w` the corresponding `M` weights. `f` is a [`QuadratureIntegrand`](@ref) carrying
the runtime-compiled integrand and bound evaluators; for every outer collocation point
it is evaluated on the `n`-times repeated outer point combined with the `M` mapped
inner nodes (the inner integration variables are appended to the network input), and
the weighted sum over the `M` inner nodes gives one value per outer point, returned as
a `1 × n` row.

Because the node set is fixed, the whole term is a static array expression that
Reactant can compile and differentiate; adaptive quadrature is not supported.
"""
function quadrature end
Symbolics.@register_array_symbolic quadrature(
    f::Any, X::AbstractMatrix, θ::AbstractVector, ξ::AbstractMatrix, w::AbstractVector
) begin
    size = (1, size(X, 2))
    eltype = Real
end false
SymbolicUtils.promote_symtype(::typeof(quadrature), f, X, θ, ξ, w) = Array{Real, 2}
function SymbolicUtils.promote_shape(
        ::typeof(quadrature), shf::SymbolicUtils.ShapeT, shX::SymbolicUtils.ShapeT,
        shθ::SymbolicUtils.ShapeT, shξ::SymbolicUtils.ShapeT, sh_w::SymbolicUtils.ShapeT
    )
    shX isa SymbolicUtils.Unknown && return SymbolicUtils.Unknown(2)
    return SymbolicUtils.ShapeVecT([1:1, 1:length(shX[2])])
end

"""
    default_adtype()

The automatic differentiation backend `discretize` uses unless `adtype` is given:
`AutoReactant()` when OptimizationReactant.jl is loaded in the session
(`using OptimizationReactant`), which compiles the objective, gradient and
value-and-gradient evaluation through Reactant and differentiates them with Enzyme
inside the compiled program, and `AutoEnzyme()` otherwise (reverse mode, static
activity analysis). The generated objective passes static activity analysis, so
runtime activity is not needed; under `Enzyme.set_runtime_activity` the reverse
pass through an `additional_loss` closure was observed to overwrite arrays the
closure captures. `AutoZygote()` is the recommended fallback for `additional_loss`
closures that mutate captured state, and remains selectable through `adtype`.
"""
function default_adtype()
    if Base.get_extension(@__MODULE__, :NeuralPDEOptimizationReactantExt) === nothing
        return AutoEnzyme()
    end
    return AutoReactant()
end

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
* `integral_alg`: the fixed-node quadrature rule used to lower `Integral` terms.
  Must be `Integrals.GaussLegendre`; defaults to `GaussLegendre()`, or to the
  `quadrature_alg` of `strategy` when it is a [`QuadratureTraining`](@ref).
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
    integral_alg
    eval_points::Int
end

function PhysicsInformedNN(
        chain, strategy::AbstractTrainingStrategy; init_params = nothing,
        rng::AbstractRNG = Random.default_rng(), derivative = FiniteDifferenceDerivative(),
        param_estim::Bool = false, additional_loss = nothing,
        boundary_policy::Symbol = :penalty, integral_alg = nothing, eval_points::Int = 100
    )
    boundary_policy in (:penalty, :constraints) || throw(
        ArgumentError(
            "`boundary_policy` must be `:penalty` or `:constraints`, got `$(boundary_policy)`."
        )
    )
    _check_integral_alg(integral_alg)
    chain = chain isa AbstractArray ? map(_to_lux, chain) : _to_lux(chain)
    return PhysicsInformedNN(
        chain, strategy, init_params, rng, derivative, param_estim, additional_loss,
        boundary_policy, integral_alg, eval_points
    )
end

_check_integral_alg(::Nothing) = nothing
function _check_integral_alg(alg)
    alg isa GaussLegendre || throw(
        ArgumentError(
            "`integral_alg` must be a fixed-node rule; only `Integrals.GaussLegendre` \
            is supported, got `$(alg)`."
        )
    )
    return nothing
end

function _integral_alg(disc::PhysicsInformedNN)
    disc.integral_alg === nothing || return disc.integral_alg
    disc.strategy isa QuadratureTraining && return disc.strategy.quadrature_alg
    return GaussLegendre()
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
    ArgumentGroup

One argument of a dependent variable in the packed network input `[arg₁; vec(arg₂); …]`.

`symbol` is the argument as declared (`t`, or the array `x`). `components` is the
flat list of scalar symbols occupying the packed slots, in column-major `vec` order.
`array` is true for an array argument, including a length-1 array such as `x[1:1]`.
A scalar argument has `array == false` and a single component, itself.
"""
struct ArgumentGroup
    symbol::Any
    components::Vector{Any}
    array::Bool
end

"""
    TrialNetwork

The symbolic neural network representing one dependent variable.

# Fields

* `depvar`: the dependent variable operation (e.g. `u` for `u(x, t)`).
* `args`: the independent variables the dependent variable is called with, after
  array arguments have been packed into scalar components.
* `NN`: the symbolic network callable parameter (see `ModelingToolkitNeuralNets`).
* `θ`: the array unknown holding the flat network parameters.
* `output`: the output row of `NN` used for this dependent variable.
* `noutputs`: the number of outputs of `NN`.
* `chain`: the Lux layer.
* `groups`: the `ArgumentGroup`s of the declared call when an argument is an
  array, or `nothing` when every argument is scalar. The groups invert the
  packing: slot `k` of the network input is component `k` of `[arg₁; vec(arg₂); …]`.
* `original`: the declared dependent-variable call before packing (`u(t, x)`), or
  `nothing` when `groups` is `nothing`.
* `expanded`: the packed call (`u(t, x[1], …)`), or `nothing` when `groups` is
  `nothing`.
"""
struct TrialNetwork{D, A, N, P, C, G, O, E}
    depvar::D
    args::A
    NN::N
    θ::P
    output::Int
    noutputs::Int
    chain::C
    groups::G
    original::O
    expanded::E

    function TrialNetwork(
            depvar, args, NN, θ, output::Int, noutputs::Int, chain,
            groups = nothing, original = nothing, expanded = nothing
        )
        return new{
            typeof(depvar), typeof(args), typeof(NN), typeof(θ), typeof(chain),
            typeof(groups), typeof(original), typeof(expanded),
        }(depvar, args, NN, θ, output, noutputs, chain, groups, original, expanded)
    end
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
* `extra_params`: non-tunable parameters created while lowering `Integral` terms
  (the `QuadratureIntegrand` callable, quadrature nodes and weights of each integral).
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
    extra_params::Vector{Any}
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
