# Physics-Informed Neural Networks

Using PINNs solver, we might solve general nonlinear PDEs:

![generalPDE](https://user-images.githubusercontent.com/12683885/86625781-5648c800-bfce-11ea-9d99-fbcb5c37fe0c.png)

 with suitable boundary conditions:

 ![bcs](https://user-images.githubusercontent.com/12683885/86625874-8001ef00-bfce-11ea-9417-1a216c7d90aa.png)

Where time t is as a special component of x, and Ω contains the temporal domain.

We describe the PDE in the form of the ModelingToolKit interface. See an example of how this can be done above or take a look at the tests.

General PDE Problem can be defined using a `PDESystem`:

```julia
PDESystem(eq,bcs,domains,param,var)
```

Here, `eq` is equation, `bcs` is boundary conditions, `param` is parameter of eqution (like `[x,y]`) and var is varibles (like `[u]`).

The method `discretize` do interpret from ModelingToolkit PDE form to the PINNs Problem.

```julia
discretize(pde_system, discretization)
```

To solve this problem use `NNDE` algorithm.

```julia
NNDE(chain,opt, autodiff=false)
```

Here, `chain` is a Flux.jl chain with d dimensional input and 1 dimensional output. `opt` is a Flux.jl optimizer. And `autodiff` is a boolean variable that determines whether to use automatic differentiation(not supported while) or numerical.
