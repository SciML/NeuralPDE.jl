# Poisson Equation

In this example, we demonstrate how `NeuralPDE.jl` can be used to solve partial differential equations on complex geometries. We solve the poisson equation on an annulus domain.

```math
\begin{align*}
-(∂^2_x + ∂^2_y)u &= 1 \, (x,y)\in\Omega, \\
\Omega &= \{(x,y) | 0.5 \leq x^2 + y^2 \leq 1.0 \}
\end{align*}
```

We represent *physical* coordinates, ``(x,y)``, and field variable ``u`` in terms of reference coordinates ``r,\theta`` which stand for *radius*, and *angle* respectively. We apply the following boundary conditions:

```math
\begin{align*}
u|_{r=0.5} = u|_{r=1.0} = 0
\end{align*}
```

## Copy-Pastable Code

```julia
```

## Coordinate Transform and Differential Operators

We represent the field vairable, ``u``, and physical coordinates ``x,y`` in terms of reference variables ``r,\theta``:

```math
u(r,\theta), \, x(r,\theta), \, y(r,\theta)
```

```julia
```

To obtain derivateives with respect to ``x`` and ``y``, we employ the chain rule:

```math
\begin{align}
\partial_x u(r,s) &= u_r(r,s)\partial_x r + u_s(x,y)\partial_x s\\
\partial_y u(r,s) &= u_r(r,s)\partial_y r + u_s(x,y)\partial_y s
\end{align}

\implies
\begin{bmatrix} \partial_x \\ \partial_y \end{bmatrix} u(r,s)
=
\begin{bmatrix} r_x & \theta_x\\ r_y & \theta_y \end{bmatrix}
\begin{bmatrix} \partial_r \\ \partial_s \end{bmatrix} u(r,s)
```

To take gradients with respect to ``x,y``, we need to find ``r_x,\, r_y,\, \theta_x,\, \theta_y``. We begin by observing that
```math
\begin{align}
\partial_x x(r,s) &= 1,\\
\partial_y x(r,s) &= 0,\\
\partial_x y(r,s) &= 0,\\
\partial_y y(r,s) &= 1
\end{align}

\implies
\begin{bmatrix} 1 & 0\\ 0 & 1 \end{bmatrix}
=
\begin{bmatrix} r_x & \theta_x\\ r_y & \theta_y \end{bmatrix}
\begin{bmatrix} x_r & y_r\\ x_\theta & y_\theta \end{bmatrix}

\implies
\begin{bmatrix} r_x & \theta_x\\ r_y & \theta_y \end{bmatrix}
=
(1/J)
\begin{bmatrix} y_\theta & -y_r\\ -x_\theta & x_r \end{bmatrix}
```
The gradients are implemented as follows:
```julia
```

The second derivaties are obtained by by composing first derivaties:

```julia
```

![poisson_annulus](https://user-images.githubusercontent.com/36345239/128362706-b39a6160-370c-43b1-b939-46214e5c3730.png)
