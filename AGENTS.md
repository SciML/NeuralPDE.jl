# Agent notes

## Testing

`test/runtests.jl` uses `SciMLTesting.run_tests()`; groups come from
`test/test_groups.toml` and folder groups auto-discover `test/<GROUP>/*.jl`.
Run a group with:

```bash
GROUP=<Name> julia --project=. -e 'using Pkg; Pkg.test()'
```

`QA` (Aqua + ExplicitImports) is a group like any other. Test dependencies come
from the root `[extras]`/`[targets]`; `Pkg.test()` resolves them — do not run
test files with `--project=.` directly (imports like `OptimizationOptimJL` are
not in the main env).

## Formatting and lint

- Format/check with Runic.jl: `julia -m Runic -c <paths>` (the `runic` binary on
  PATH is an unrelated tool).
- Spell check: `typos` over the changed files.

## Symbolic registration pitfalls

- `@register_array_symbolic` emits one wrapper method per symbolic/concrete
  argument combination. A runtime method on the same generic function is
  ambiguous with all of them unless it is the most general signature: define the
  runtime body on fully untyped arguments (`nn_eval(f, X, θ) = f(X, θ)` pattern)
  and validate argument types inside, or Aqua reports hundreds of ambiguities.
- `DomainSets.endpoints` is owned by `IntervalSets` and fails `ExplicitImports`
  when accessed as `DomainSets.endpoints`. `DomainSets.infimum`/`supremum` pass
  that check but do an `a > b` emptiness test that throws on symbolic bounds.
  Read `d.left`/`d.right` on `TypedEndpointsInterval` domains instead.

## Device-genericity pitfalls

- The generated objective must keep every reduction behind a registered
  function (`_mean_square`, `_weighted_square_sum`) with a `ChainRulesCore.rrule`.
  A literal `sum`/`mapreduce` over a GPU array hits `GPUArraysCore._mapreduce` →
  `task_local_storage`, which Zygote cannot differentiate. Test devices with
  JLArrays under `JLArrays.allowscalar(false)`, and check lowered residuals for
  host array constants: JLArrays accepts mixed host/device arithmetic that CUDA
  rejects. CUDA tests are required before claiming GPU support.
- `remake(prob; u0 = dev(prob.u0), p = [block.xs => dev(X), ...])` moves the
  collocation data and network parameters; `resample!` must write device arrays (`similar` + `copyto!`),
  and `PDENoTimeSolution` copies `θ` back to the host with `Array`.
- Device tests and examples must select `AutoZygote()` explicitly and exercise
  `solve`, resampling, and host solution evaluation. Preserve the discretization
  element type during transfer: finite-difference steps are fixed at lowering time.
