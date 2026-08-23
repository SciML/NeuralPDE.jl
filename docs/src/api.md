# API

NeuralPDE's own public API — the solver algorithms (`NNODE`, `NNDAE`, `BNNODE`,
`PINOODE`, `NNSDE`, `SDEPINN`, `DeepGalerkin`), the `PhysicsInformedNN` discretization,
the training strategies and the adaptive loss functions — is documented
in the Manual: [ODE solvers](manual/ode.md), [DAE solvers](manual/dae.md),
[PINN discretizations](manual/pinns.md), [Bayesian PINNs](manual/bpinns.md),
[physics-informed neural operators](manual/pino_ode.md) and
[neural adapters](manual/neural_adapters.md).

This page documents the *reexported* surface instead: names that `using NeuralPDE`
brings into scope but that another package owns and documents.

## Reexported symbolic front end

Writing down a `PDESystem` is the normal documented entry point to NeuralPDE, so
`using NeuralPDE` also brings in the symbolic front end needed to express one. These
names are owned and documented by
[ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/stable/) (several of them
re-exported by ModelingToolkit from
[Symbolics](https://docs.sciml.ai/Symbolics/stable/)):

  - Declaring symbols: `@parameters`, `@variables`, `@named`, `@register_symbolic`
  - Systems: `PDESystem`, `mtkcompile`, `@mtkcompile`, `unknowns`
  - Operators: `Differential`, `Integral`
  - Domain endpoints: `infimum`, `supremum`
  - The `ModelingToolkit` module itself

Anything else from ModelingToolkit or Symbolics must be imported from that package
directly — for example `System`, `equations`, `parameters` and `observed` are
system-manipulation API that NeuralPDE does not use in its own documented workflow, so
they are deliberately not reexported. Interval domains still come from
[DomainSets](https://github.com/JuliaApproximation/DomainSets.jl)
(`using DomainSets: Interval`), which NeuralPDE does not reexport.

## Reexported SciML common interface

`using NeuralPDE` also brings in the parts of the SciML common interface needed to
build a problem, solve it and inspect the result, so they do not have to be imported
separately. These names are owned and documented by
[SciMLBase](https://docs.sciml.ai/SciMLBase/stable/):

  - Problems: `ODEProblem`, `DAEProblem`, `SDEProblem`, `NoiseProblem`,
    `OptimizationProblem`
  - Functions: `ODEFunction`, `ODEInputFunction`, `OptimizationFunction`
  - Solutions: `ODESolution`, `PDETimeSeriesSolution`
  - Solving and discretizing: `solve`, `init`, `remake`, `discretize`,
    `symbolic_discretize`
  - Return status: `ReturnCode`
  - The `SciMLBase` module itself

Anything else from SciMLBase must be imported from SciMLBase directly. In particular
the ensemble interface, the callback types and the integrator interface are not
reexported: NeuralPDE's solvers train a network rather than step an integrator, so
those names are not part of its documented use.

!!! note
    
    This list is kept in sync in three places: the reexport `export` blocks in
    `src/NeuralPDE.jl`, the `REEXPORTS` tuple in `test/qa/qa.jl` (which is what
    `run_qa`'s `reexports_allow` is given, and which a test checks against
    `names(NeuralPDE)`), and this page.
