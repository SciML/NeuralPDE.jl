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

For constrained-solver regression tests, assert the raw solver status and bound the
measured constraint violation by the configured constraint tolerance. OptimizationIpopt
maps acceptable-level termination to `ReturnCode.Success`; `acceptable_iter = 0`
does not rule out every such exit. Converge both policies before comparing their errors.

Fixed Lux state regressions must compare single-point, batched, and batch-position
values, and check stored state before and after default-backend differentiation.
Deterministic losses alone do not establish pointwise field semantics. State alias
regressions must mix floating precisions and reach shared buffers through `Ref` and
array containers; required keys also need coverage with empty tuple values.

Each folder group runs all its files in one Julia process. The `arch = "x86"`
lanes have a 32-bit address space, so compile-heavy files (Enzyme derivative
tests) live in their own group (`InterfaceEnzyme`) rather than sharing a process
with the rest of `Interface`.

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

## Additional-loss AD

Closure field types do not determine `additional_loss` AD compatibility: callbacks
with non-`isbits` captures can still differentiate correctly with Enzyme. Choose a
backend explicitly at call sites with demonstrated requirements, and retain
captured-metadata, buffer-writing callbacks in default-backend regression coverage.
