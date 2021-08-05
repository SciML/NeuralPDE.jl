# Introduction

State of the art methodologies for solving partial differential equations are predicated on robust function approximation frameworks, and the ability to do calculus (differentiation/integration) on said function representations. For example, inite element discretizations are based on piecewise continuous polynomials, and spectral methods are based on a global fourier/polynomial representation.

This work is aimed at developing a robust and highly accurate (spectral convergence type) function approximation scheme is needed to solve PDEs using differentiable programming techniques. We propose leranable models for approximating C⁰ functions in space, Rⁿ by extending of high-order Finite Element Mehtods to a differentiable programming framework.

## Element Based Methods

Element based methods tessalate or partition the computational domain Ω ⊂ Rⁿ in a preprocessing step called "meshing". Each mesh "element" (Ωᵢ ⊂ Ω) is spanned by order "n" lagrange polynomials over Gauss Lobatto or Chebychev integration points (in each space direction). These polynomials form a basis for approximating functions within the element. The approximation is (usually) spectrally convergent, why is why some implementation of high order pFEM are also referred to as Spectral Element Method.

The function approximation is *local* to the elemnt, meaning the basis functions vanish outside the element. Think of the approximating functions as being multiplied by the indicator function of the element they belong to. In solving PDEs, one writes out linear system for the scaling coefficients of the basis functions. Element based approximations yields tight coupling within each element, while C⁰ continuity is maintained across element boundaries by *gather-scatter* operations.

## Complex Geometry

Element based methods represent complex geometry by tesselating the domain (meshing) into n-dimensinoal simplices (eg hexagons). Each *deformed* simplex is mapped to a reference element based on a unique coordinate transformation. Physical coordinates, `x,y` are represented as functions of the reference coordinates `r,s`, like velocity, pressure or other field variables. `x(r,s), y(r,s)`. Such a formulation is called an isoparametric mapping. SEM works well when each element is only a "little bit" deformed, i.e. for each element, `x(r,s), y(r,s)` can be represented well by the approximation basis.

As geometries can't apriori be broken down into elements, the process of meshing is time, and compute expensive. Further, the fidelity of the solution, and time-to-solution depend heavily on the quality of the mesh, and human input is frequently required in cleaning/fixing meshes. All this makes meshing and mesh iterations a bottleneck in the engineering design cycle.


# Method

## Trainable Mesh

A mesh is simply disjoint indicator functions of subsets to the domain.

$$ \Omega_i(x) = \begin{cases} 1,& x \in \Omega_i\\
                                            0,& x \in \Omega
                              \end{cases}
$$

[POUnets](https://arxiv.org/abs/2101.11256) extended the idea of a mesh to a general Partition of Unity (POU). A partition of unity is a set of functions $\phi_i$ such that for all points $x$ in the domain, they sum to unity.

$$ \forall x\in\Omega, \Sigma_i \phi_i(x) = 1 $$

There are three components to a POUnet:
1. A neural network that outputs `m` partition functions (last layer of NN is `softmax`)
2. Matrix `C` of size `[m,n]` scaling coefficients
3. A set of `n` global functions that span any Banach space

Within each partition (which can be continuous and overlapping), the sum-product of scaling coefficients and function space creats an approximation to the target function. The contribution from each partition is summed to arrive at the final approximation to the target function: `\phi_i C_ij \psi_j`. The partition functions are expect to provide localization to the features of the target function. If the partition functioinss are piecewise continuous, like the output of a `relu` network, then the POUnet can approximate **piecewise continuous functions**. This allows the neural network to play to its strengths (space partitioning) instead of simultaneously partitioning space and approximating a function basis.

## Trainable Function Space

We extend POUnets by applying a unique, trainable affine transformation to the input to the function space. This was inspired by the fact that in standard FEM, the polynomial basis is defined on a canonical domain, and is transformed to physical space via an affine transformation that is element specific. In practice, we'd like a unique affine transformation for every function in the domain.

We apply the transformation `x -> @. \alpha * x + \beta`, where `\alpha`, and `\beta` are trainable arrays of size `[m,n]`, and then pass the output through a `cosine` activation `x\bar -> cos(pi*x\bar)` to achieve our function space, `\psi_ij(x) = cos(pi*(\alpha_ij * x + \beta_ij))`. We obtain the target function in a similar manner, by scaling and summing the basis functions in every partition, and then summing the contribution of every parition.

```
f_ij = C_ij .* \psi_ij
g_i = sum(f_ij, dims=1)

h_i = g_i .* \phi_i
res = sum(h_i)
```

Tuning the function approximation, and not just the scaling coefficients, within each partition seems to allow for faster training.


## Thoughts, Ideas

* initialize pou as indicator function on a computational mesh
  and make the mesh points the only trainable parameter
* let's call it MeshNet
* how do you make partition functions differentiable? use some
  dirac delta function/ heaviside function magic
* create a partition of unity using the signed distance function
  eg. clip partition functions to 0 or 1 where SDF(x⃗) = 0. read up on interface type methods (immersed boundary)
* allow partitions to collapse onto themselves
* how to enforce Cⁿ continuity in general? Enforce the same
  requirement on the partition functions?
* consider loss in frequency domain (whatever that means) could
  help in tuning α, β. train data in fourier space?
