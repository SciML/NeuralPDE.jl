# NeuralPDE.jl: Automatic Physics-Informed Neural Networks (PINNs)

[NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl) is a solver package which 
consists of neural network solvers for partial differential equations using 
physics-informed neural networks (PINNs) and the ability to generate neural
networks which both approximate physical laws and real data simultaniously.

## Features

  - Physics-Informed Neural Networks for ODE, SDE, RODE, and PDE solving.
  - Ability to define extra loss functions to mix xDE solving with data fitting (scientific machine learning).
  - Automated construction of Physics-Informed loss functions from a high-level symbolic interface.
  - Sophisticated techniques like quadrature training strategies, adaptive loss functions, and neural adapters
    to accelerate training.
  - Integrated logging suite for handling connections to TensorBoard.
  - Handling of (partial) integro-differential equations and various stochastic equations.
  - Specialized forms for solving `ODEProblem`s with neural networks.
  - Compatibility with [Flux.jl](https://docs.sciml.ai/Flux.jl/stable/) and [Lux.jl](https://docs.sciml.ai/Lux/stable/).
    for all the GPU-powered machine learning layers available from those libraries.
  - Compatibility with [NeuralOperators.jl](https://docs.sciml.ai/NeuralOperators/stable/) for
    mixing DeepONets and other neural operators (Fourier Neural Operators, Graph Neural Operators,
    etc.) with physics-informed loss functions.

## Installation

Assuming that you already have Julia correctly installed, it suffices to import
NeuralPDE.jl in the standard way:

```julia
import Pkg
Pkg.add("NeuralPDE")
```

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:
    
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Citation

If you use NeuralPDE.jl in your research, please cite [this paper](https://arxiv.org/abs/2107.09443):

```tex
@misc{https://doi.org/10.48550/arxiv.2107.09443,
  doi = {10.48550/ARXIV.2107.09443},
  url = {https://arxiv.org/abs/2107.09443},
  author = {Zubov, Kirill and McCarthy, Zoe and Ma, Yingbo and Calisto, Francesco and Pagliarino, Valerio and Azeglio, Simone and Bottero, Luca and Luján, Emmanuel and Sulzer, Valentin and Bharambe, Ashutosh and Vinchhi, Nand and Balakrishnan, Kaushik and Upadhyay, Devesh and Rackauckas, Chris},
  keywords = {Mathematical Software (cs.MS), Symbolic Computation (cs.SC), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {NeuralPDE: Automating Physics-Informed Neural Networks (PINNs) with Error Approximations},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```

## Flux.jl vs Lux.jl

Both Flux and Lux defined neural networks are supported by NeuralPDE.jl. However, Lux.jl neural networks are greatly preferred for many
correctness reasons. Particularly, a Flux `Chain` does not respect Julia's type promotion rules. This causes major problems in that
the restructuring of a Flux neural network will not respect the chosen types from the solver. Demonstration:

```julia
using Flux, Tracker
x = [0.8; 0.8]
ann = Chain(Dense(2, 10, tanh), Dense(10, 1))
p, re = Flux.destructure(ann)
z = re(Float64(p))
```

While one may think this recreates the neural network to act in `Float64` precision, [it does not](https://github.com/FluxML/Flux.jl/pull/2156)
and instead its values will silently downgrade everything to `Float32`. This is only fixed by `Chain(Dense(2, 10, tanh), Dense(10, 1)) |> f64`.
Similar cases will [lead to dropped gradients with complex numbers](https://github.com/FluxML/Optimisers.jl/issues/95). This is not an issue
with the automatic differentiation library commonly associated with Flux (Zygote.jl) but rather due to choices in the neural network library's
decision for how to approach type handling and precision. Thus when using DiffEqFlux.jl with Flux, the user must be very careful to ensure that
the precision of the arguments are correct, and anything that requires alternative types (like `TrackerAdjoint` tracked values,
`ForwardDiffSensitivity` dual numbers, and TaylorDiff.jl differentiation) are suspect.

Lux.jl has none of these issues, is simpler to work with due to the parameters in its function calls being explicit rather than implicit global
references, and achieves higher performance. It is built on the same foundations as Flux.jl, such as Zygote and NNLib, and thus it supports the
same layers underneith and calls the same kernels. The better performance comes from not having the overhead of `restructure` required.
Thus we highly recommend people use Lux instead and only use the Flux fallbacks for legacy code.

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@raw html
You can also download the 
<a href="
```

```@eval
using TOML
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
       "/assets/Manifest.toml"
```

```@raw html
">manifest</a> file and the
<a href="
```

```@eval
using TOML
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
       "/assets/Project.toml"
```

```@raw html
">project</a> file.
```
