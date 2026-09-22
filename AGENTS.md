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
