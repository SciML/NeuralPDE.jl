# NeuralPDE.jl: Scientific Machine Learning for Partial Differential Equations

[NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl)
NeuralPDE.jl is a solver package which consists of neural network solvers for
partial differential equations using physics-informed neural networks (PINNs).

## Features

- Physics-Informed Neural Networks for ODE, SDE, RODE, and PDE solving
- Ability to define extra loss functions to mix xDE solving with data fitting (scientific machine learning)
- Automated construction of Physics-Informed loss functions from a high level symbolic interface
- Sophisticated techniques like quadrature training strategies, adaptive loss functions, and neural adapters
  to accelerate training
- Integrated logging suite for handling connections to TensorBoard
- Handling of (partial) integro-differential equations and various stochastic equations
- Specialized forms for solving `ODEProblem`s with neural networks

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
