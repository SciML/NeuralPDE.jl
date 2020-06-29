# NeuralNetDiffEq

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/SciML/NeuralNetDiffEq.jl.svg?branch=master)](https://travis-ci.org/SciML/NeuralNetDiffEq.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/v0eop301bx105av4?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/neuralnetdiffeq-jl)
[![Coverage Status](https://coveralls.io/repos/JuliaDiffEq/NeuralNetDiffEq.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiffEq/NeuralNetDiffEq.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaDiffEq/NeuralNetDiffEq.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiffEq/NeuralNetDiffEq.jl?branch=master)

The repository is for the development of neural network solvers of differential equations such as physics-informed
neural networks (PINNs) and deep BSDE solvers. It utilizes techniques like deep neural networks and neural
stochastic differential equations to make it practical to solve high dimensional PDEs efficiently through the
likes of scientific machine learning (SciML).

## Installation

Open [REPL](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html)

```jl
add NeuralNetDiffEq
```

## Related Packages

- [ReservoirComputing.jl](https://github.com/MartinuzziFrancesco/ReservoirComputing.jl) has an implementation of the [Echo State Network method](https://arxiv.org/pdf/1710.07313.pdf) for learning the attractor properties of a chaotic system.
