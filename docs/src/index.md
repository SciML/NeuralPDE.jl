# NeuralPDE.jl: Scientific Machine Learning for Partial Differential Equations

[NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl)
is a solver package which consists of neural network solvers for
partial differential equations using scientific machine learning (SciML)
techniques such as physics-informed neural networks (PINNs) and deep
BSDE solvers. This package utilizes deep neural networks and
neural stochastic differential equations to solve high-dimensional PDEs
at a greatly reduced cost and greatly increased generality compared with classical methods.

## Features

- Physics-Informed Neural Networks for automated PDE solving
- Forward-Backwards Stochastic Differential Equation (FBSDE) methods for parabolic PDEs
- Deep-learning-based solvers for optimal stopping time and Kolmogorov backwards equations

## Citation

If you use NeuralPDE.jl in your research, please cite [this paper](https://arxiv.org/abs/2107.09443):

```tex
@article{zubov2021neuralpde,
  title={NeuralPDE: Automating Physics-Informed Neural Networks (PINNs) with Error Approximations},
  author={Zubov, Kirill and McCarthy, Zoe and Ma, Yingbo and Calisto, Francesco and Pagliarino, Valerio and Azeglio, Simone and Bottero, Luca and Luj{\'a}n, Emmanuel and Sulzer, Valentin and Bharambe, Ashutosh and others},
  journal={arXiv preprint arXiv:2107.09443},
  year={2021}
}
```
