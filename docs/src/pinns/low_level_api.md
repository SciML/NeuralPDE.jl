### Low-level API
TODO

These additional methods exist to help with introspection:

- `symbolic_discretize(pde_system,discretization)`: This method is the same as `discretize` but instead
  returns the unevaluated Julia function to allow the user to see the generated training code.

  Keyword arguments:

  - `pde_system`
  - `discretization`

  ```julia

  ```

- `build_symbolic_loss_function(eqs,indvars,depvars, phi, derivative,initθ; bc_indvars=nothing)`: return symbol

Keyword arguments:

- `eqs`,
- `indvars,depvars`,
- `phi`,
- `derivative`,
- `initθ`,
- `bc_indvars`.


```julia
```

- `build_symbolic_equation(eq,indvars,depvars)`:

```julia
```

- `build_loss_function(eqs, indvars, depvars, phi, derivative, initθ; bc_indvars=nothing)`:

- `get_loss_function(loss_functions, train_sets, strategy::)`:

- `get_phi(chain)`:

- `get_numeric_derivative()`:

- `generate_training_sets(domains,dx,bcs,_indvars::Array,_depvars::Array)`:

- `get_bc_varibles(bcs,_indvars::Array,_depvars::Array)`:

- `get_bounds()`:


See how this can be used in the docs examples or take a look at the tests.
