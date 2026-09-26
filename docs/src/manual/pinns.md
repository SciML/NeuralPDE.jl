# `PhysicsInformedNN` Discretizer for PDESystems

Using the PINNs solver, we can solve general nonlinear PDEs:

```math
f{\left(x; \frac{∂u}{∂x_1}, \dots, \frac{∂u}{∂x_d}; \frac{∂^2 u}{∂x_1 ∂x_1}, \frac{∂^2 u}{∂x_1 ∂x_d}; \dots ; \lambda\right)} = 0, x \in \Omega,
```

with suitable boundary conditions:

```math
B(u, x) = 0 \; \text{ on } \; ∂\Omega
```

where time t is a special component of x, and Ω contains the temporal domain.

PDEs are defined using the ModelingToolkit.jl `PDESystem`:

```julia
@named pde_system = PDESystem(eq, bcs, domains, param, var)
```

Here, `eq` is the equation, `bcs` represents the boundary conditions, `param` is
the parameter of the equation (like `[x,y]`), and `var` represents variables (like `[u]`).
For more information, see the
[ModelingToolkit.jl PDESystem documentation](https://docs.sciml.ai/ModelingToolkit/stable/API/PDESystem/).

## The pipeline

`PhysicsInformedNN` follows the same `PDESystem → System → problem → solution` pipeline
as the other SciML discretizers (for example MethodOfLines.jl):

1. `symbolic_discretize(pdesys, discretization)` returns a `ModelingToolkit.System`. Each
   dependent variable is represented by a `ModelingToolkitNeuralNets.SymbolicNeuralNetwork`
   whose parameters are array unknowns of the system; the residual of every equation and
   boundary condition is a symbolic array expression over a matrix of collocation points
   stored as a parameter of the system; each residual contributes one cost (or, for
   boundary conditions with `boundary_policy = :constraints`, a pointwise equality).
2. `discretize(pdesys, discretization)` samples the collocation points with the training
   strategy and generates an `OptimizationProblem` through ModelingToolkit's
   `OptimizationProblem(sys, op)` constructor. Keyword arguments such as `adtype`
   (automatic differentiation backend) and `weights` (weighted sum of the costs) are
   forwarded to it. The system is only `complete`d, not passed through `mtkcompile`:
   the network parameters stay array unknowns of the problem, which keeps the generated
   code independent of the number of parameters.
3. `solve(prob, optimizer)` returns a `PDENoTimeSolution` which evaluates the trained
   networks: `sol[u(x, t)]` on the evaluation grid, `sol(x, t; dv = u(x, t))` at arbitrary
   points, and `sol.original_sol` for the underlying `OptimizationSolution`.

## Exact boundary constraints

By default, boundary residuals contribute mean squared penalty costs. Set
`boundary_policy = :constraints` to impose each boundary residual as an equality at its
collocation points. The selected optimization solver must support equality constraints;
for example, `IpoptOptimizer` from OptimizationIpopt does. Constraint derivatives for
these array residuals need an automatic differentiation backend, so pass `adtype` when
discretizing.

The following 2D Poisson problem has the solution
`u(x, y) = sinpi(x) * sinpi(y) / (2pi^2)`. A fixed polynomial feature layer and a
trainable linear output layer give a biquadratic trial space. Both optimization
problems are convex quadratics, but this space cannot represent the sinusoidal
solution exactly. The same seed gives both policies the same initial weights and
collocation points; the points stay fixed during each solve.

```@example boundary_constraints
using NeuralPDE, ModelingToolkit, Lux, Random, SciMLBase
using DomainSets: Interval
using OptimizationIpopt
using ADTypes: AutoForwardDiff
import Ipopt

@parameters x y
@variables u(..)
Dxx, Dyy = Differential(x)^2, Differential(y)^2
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
@named poisson = PDESystem(eq, bcs,
    [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)], [x, y], [u(x, y)])

legendre(z) = (one.(z), 2z .- 1, 6z .^ 2 .- 6z .+ 1)
features(X) = vcat([
    a .* b for a in legendre(X[1:1, :]) for b in legendre(X[2:2, :])
]...)
chain = Chain(WrappedFunction(features), Dense(9, 1; use_bias = false))
problem(policy) = discretize(poisson,
    PhysicsInformedNN(chain, StochasticTraining(100; bcs_points = 2);
        rng = Xoshiro(1173), boundary_policy = policy);
    adtype = AutoForwardDiff())
penalty_prob = problem(:penalty)
constrained_prob = problem(:constraints)
constr_viol_tol = 1.0e-9
optimizer = IpoptOptimizer(; acceptable_iter = 0, constr_viol_tol,
    nlp_scaling_method = "none")

results = map((:penalty, :constraints), (penalty_prob, constrained_prob)) do policy, prob
    sol = solve(prob, optimizer; maxiters = 100, reltol = 1.0e-8, verbose = 0)
    status = Ipopt.ApplicationReturnStatus(sol.original_sol.original.status)
    @assert status == Ipopt.Solve_Succeeded
    residual = zeros(length(constrained_prob.lcons))
    constrained_prob.f.cons(residual, sol.original_sol.u, constrained_prob.p)
    boundary_error = maximum(abs, residual)
    interior_error = maximum(
        abs(sol(a, b; dv = u(x, y)) - sinpi(a) * sinpi(b) / (2pi^2))
        for a in 0.05:0.05:0.95, b in 0.05:0.05:0.95)
    (; policy, status, boundary_error, interior_error)
end
@assert results[2].boundary_error <= constr_viol_tol
@assert results[1].boundary_error > constr_viol_tol
@assert results[2].interior_error < results[1].interior_error
results
```

Install OptimizationIpopt in the environment when using this solver; it brings the Ipopt
binary, which is distributed under the Eclipse Public License 2.0.

Ipopt's `acceptable_iter = 0` disables the heuristic based on consecutive
acceptable iterates. Other termination paths can still return
`Solved_To_Acceptable_Level`, and OptimizationIpopt maps that status to
`ReturnCode.Success`. To require strict convergence, check
`Ipopt.ApplicationReturnStatus(sol.original_sol.original.status) == Ipopt.Solve_Succeeded`
and evaluate the constraint residual against the configured `constr_viol_tol`.
See [Ipopt's termination options](https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_Termination).

Exact constraints hold at the boundary collocation points. They do not guarantee
accuracy between those points or in the interior. Nonlinear networks can also have
poor local optima; a penalty solution can provide an initial guess through
`remake(prob; u0 = penalty_sol.original_sol.u)`, but both feasibility and interior
accuracy still need checking. A finite penalty can trade boundary error for a smaller
PDE residual when the trial space cannot represent the PDE solution. If both residuals
can vanish simultaneously, the two policies can share a minimizer.

## Spatial derivatives with Enzyme

Select `derivative = EnzymeForwardDerivative()` when constructing
`PhysicsInformedNN` to evaluate spatial derivatives with Enzyme forward mode:

```julia
disc = PhysicsInformedNN(chain, GridTraining(0.1);
    derivative = EnzymeForwardDerivative(), rng = Xoshiro(1))
prob = discretize(pde_system, disc; adtype = AutoEnzyme())
```

This choice controls differentiation with respect to network inputs. The separate
`adtype` controls differentiation of the training objective with respect to network
weights. Both `AutoEnzyme()` and `AutoZygote()` can differentiate the spatial JVPs;
the Zygote path uses a ChainRules pullback evaluated by Enzyme reverse mode.

Pure derivatives of orders one through four and mixed derivatives of total order
at most four use nested forward-over-forward differentiation. Each level seeds
the requested input coordinate across all collocation columns. This avoids
constructing a full Jacobian and uses no finite-difference step size. Nesting
also allows different coordinate directions at successive levels. Higher orders
increase compilation and evaluation costs, so benchmark the backend on your network.

Apply `Differential` operators to dependent-variable calls whose arguments are
independent variables or numeric literals; derivative boundary conditions such as
`Differential(x)(u(1.0))` are supported. Networks must process columns independently
and support Enzyme. Derivatives of composite expressions or transformed network
arguments are not supported by this backend; write the product or chain rule
explicitly in the PDE. `FiniteDifferenceDerivative()` remains the default.

## Integral terms

`Symbolics.Integral` terms are supported: they lower to a fixed-node quadrature over
the batch of collocation points. For

```julia
@parameters t τ
@variables u(..)
I = Integral(τ in DomainSets.ClosedInterval(0.0, t))
@named sys = PDESystem([I(u(τ)) ~ t^2 / 2], [u(0.0) ~ 0.0],
    [t ∈ Interval(0.0, 1.0)], [t], [u(t)])
```

the integrating variable `τ` is declared with `@parameters` like any other symbol and
needs no domain of its own; the bounds may depend on the free independent variables
(`0.0 .. t`), on earlier integrating variables for multidimensional domains, on
`PDESystem` parameters, or be infinite (`-Inf`/`Inf` bounds are mapped to a finite
`σ` interval exactly as in NeuralPDE 6). Inside the integrand the integrating
variables are appended to the network input, so shifted arguments (`u(t - τ)`),
inner derivatives (`Differential(τ)(u(τ))`) and nested `Integral` terms work as well.

The quadrature rule is fixed-node only so the lowered term is a static array
expression (Reactant compatible); it defaults to `Integrals.GaussLegendre`, inherits
`quadrature_alg` from `QuadratureTraining`, and can be set explicitly with the
`integral_alg` keyword of `PhysicsInformedNN`.

## Array arguments

A dependent variable can take an array of independent variables as one argument.
With `@parameters t x[1:d]` the call `u(t, x)` is a function of `1 + d` scalars: the
network input is the packed vector `[t; vec(x)]`, in column-major order. For
`x[1:2,1:2]` that order is `x[1,1], x[2,1], x[1,2], x[2,2]`. `Differential(x[i])`
or `Differential(x[i,j])` differentiates along that slot. A length-1 array such as
`x[1:1]` is still an array argument: pass a length-1 vector, `sol(t, [x1]; dv = u(t, x))`.
The declared grouping is kept, so the packing is reversible.

Domains are either one interval per component, `x[i] ∈ Interval(a, b)`, or a single
product domain for the array (one factor per component, in `vec` order). A boundary
on which one component is fixed is an ordinary vector argument, for example
`u(t, [0.0, x[2]])`. A `GridTraining` spacing vector, when it is not a single
number, has one entry per packed scalar in the order of the independent-variable
list after that expansion.

`sol[u(t, x)]` is the tensor product of the component evaluation grids, with shape
`(n_t, n_x₁, …, n_x_d)`. `sol(t, xvec; dv = u(t, x))` evaluates a scalar `t` and one
length-`d` vector `xvec` (a matrix with `d` rows is a list of such vectors, one per
column). The same value is `sol(t, x[1], …, x[d]; dv = u(t, x[1], …))`.

Symbolics does not ship gradient, divergence, Laplacian or curl operators. The open
pull request [JuliaSymbolics/Symbolics.jl#942](https://github.com/JuliaSymbolics/Symbolics.jl/pull/942)
is not part of the Symbolics version NeuralPDE depends on, so those operators are
written componentwise, including `Differential.(collect(x))`.

## The `PhysicsInformedNN` Discretizer

```@docs
NeuralPDE.PhysicsInformedNN
NeuralPDE.FiniteDifferenceDerivative
NeuralPDE.EnzymeForwardDerivative
SciMLBase.discretize(::PDESystem, ::NeuralPDE.PhysicsInformedNN)
NeuralPDE.default_adtype
```

## `symbolic_discretize` and the generated `System`

```@docs
SciMLBase.symbolic_discretize(::PDESystem, ::NeuralPDE.PhysicsInformedNN)
NeuralPDE.pinn_metadata
NeuralPDE.PINNMetadata
NeuralPDE.TrialNetwork
NeuralPDE.ResidualBlock
NeuralPDE.nn_eval
NeuralPDE.nn_eval_row
NeuralPDE.nn_jvp
NeuralPDE.nn_vcat
NeuralPDE.nn_veccat
NeuralPDE.quadrature
NeuralPDE.QuadratureIntegrand
```

## Solutions, resampling and transfer learning

```@docs
SciMLBase.PDENoTimeSolution(::SciMLBase.AbstractOptimizationSolution, ::NeuralPDE.PINNMetadata)
NeuralPDE.trial_function
NeuralPDE.resample!
NeuralPDE.distill
```

Warm-starting a problem from previously trained weights (transfer learning) is
`remake(prob; u0 = trained_weights)`, and replacing the collocation points of a residual
block is `remake(prob; p = [block.xs => new_points])`. Moving a trained solution into
a different network architecture is [`distill`](@ref), whose distillation recipe is
worked out in the [transfer-learning tutorial](../tutorials/transfer_learning.md).

## SDE Solvers

```@docs
NeuralPDE.NNSDE
NeuralPDE.SDEPINN
```
