# NeuralNetDiffEq

NeuralNetDiffEq.jl , which consists neural network solvers for differential equations such as physics-informed neural networks (PINNs) and deep BSDE solvers, is a package of scientific machine learning (SciML).
It utilizes deep neural networks and neural stochastic differential equations to solve high dimensional PDEs.

## Related Packages

- [ReservoirComputing.jl](https://github.com/MartinuzziFrancesco/ReservoirComputing.jl) has an implementation of the [Echo State Network method](https://arxiv.org/pdf/1710.07313.pdf) for learning the attractor properties of a chaotic system.

## Citation

If you use NeuralNetDiffEq.jl or are influenced by it's ideas for expanding it, please cite:

```latex
@article{DifferentialEquations.jl-2017,
 author = {Rackauckas, Christopher and Nie, Qing},
 doi = {10.5334/jors.151},
 journal = {The Journal of Open Research Software},
 keywords = {Applied Mathematics},
 note = {Exported from https://app.dimensions.ai on 2019/05/05},
 number = {1},
 pages = {},
 title = {DifferentialEquations.jl – A Performant and Feature-Rich Ecosystem for Solving Differential Equations in Julia},
 url = {https://app.dimensions.ai/details/publication/pub.1085583166 and http://openresearchsoftware.metajnl.com/articles/10.5334/jors.151/galley/245/download/},
 volume = {5},
 year = {2017}
}
```
