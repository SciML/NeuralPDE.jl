# Using GPU for Kolmogorov Equations
For running Kolmogorov Equations on a GPU there are somethings that are needed to be taken care of :

Convert the model parameters to `CuArrays` using the `fmap` function given by Flux.jl
```julia
m = Chain(Dense(1, 64, σ), Dense(64, 64, σ) , Dense(5, 2))
m = fmap(cu, m)
```
Unlike other solver we need to specify explicitly to the solver to run on the GPU. This can be done by passing the `use_gpu = true`  into the solver.
```julia
solve(prob, NeuralNetDiffEq.NNKolmogorov(m, opt, sdealg, ensemblealg), use_gpu = true,  verbose = true, dt = dt, dx = dx , trajectories = trajectories , abstol=1e-6, maxiters = 1000)
```
