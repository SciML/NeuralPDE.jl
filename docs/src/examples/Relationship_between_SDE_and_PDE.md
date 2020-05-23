## Stochastic Differential Equations, Deep Learning, and High-Dimensional PDEs

Now we will suss out the relationship between SDEs and PDEs and how this is used in Scientific Machine Learning to solve previously unsolvable problems with a neural network as the intermediate.

## What is an SDE?

The easiest way to understand an SDE is by looking at it computationally. For the SDE defined as:

```math
dX_t = f(X_t,t)dt + g(X_t,t)dW_t
```

the **"Euler method for SDEs"**, also known as **Euler-Maruyama**, is given by:

```math
X_n+1 = X_n + f(X_n,t_n)h + sqrt(h)g(X_n,t_n)\zeta
```

where ``\zeta ~ N(0,1)`` is a standard normal random variable ``(randn())``. Thus a stochastic differential equation is an ordinary differential equation with a small stochastic perturbation at every point in time (continuously!).

There are many definitions of **Brownian Motion**. One of the easiest to picture is that it's the random walk where at every infinitesimal ``dt`` you move by ``N(0,dt)`` (this is formally true in non-standard analysis, see **Radically Elementary Probability Theory** for details). Another way to think of it is as a limit of standard random walks. Let's say you have a random walk were every time step of ``h`` you move ``\Delta x`` to the right or left. If you let ``h/\Delta x = 1`` and send ``h \rightarrow 0``, then the limiting process is a **Brownian Motion**. All of the stuff about normal distributions just comes from the central limit theorem and the fact that you're doing infinitely many movements per second!

## Wiener Process Calculus Summarized

Brownian motion is non-differentiable, and this is fairly straightforward to prove. Take any finite `` h > 0 ``. In that interval, the probability that the Brownian motion has a given value is ``N(0,h)``. Thus it has a positive probability of being positive and a positive probability of being negative. Thus take the sequence ``h/2, h/4, `` etc. to have infinetly many subintervals. With probability 1 at least one of those values will be positive and at least one will be negative, and so for every ``h`` there is a positive or negative value, which means **Brownian Motion** is always neither increasing or decreasing. This means that, if the classical derivative existed, it must be zero, which means it's a constant function, but it's not, so by contradiction Brownian motion cannot have a classical derivative!

However, with a bunch of analysis one can derive an appropriate calculus on SDEs. Take the SDE

```Math
dx = f(x,t)dt + \sum_{i=1}^{n}g_{i}(x,t)dW_{i}
```

where ``W_{i}(t)`` is a standard Brownian motion. It0's Rules (the stochastic Chain) rule could be interpreted as:
```math
dt\times dt = 0
```

```math
dW_{i}\times dt = 0
```

```math
dW_{i}\times dW{i} = dt
```

```math
dW_{i}\times dW_{j} = 0\,\,\,\,j\neq i
```

and thus if ``y = \psi(x,t)``, Ito's rules can be written as:

```math
dy=\frac{\partial\psi}{\partial t}dt+\frac{\partial\psi}{\partial x}dx+\frac{1}{2}\frac{\partial^{2}\psi}{\partial x^{2}}\left(dx\right)^{2}
```

where, if we plug in ``dx``, we get

```math
dy=d\psi(x,t)=\left(\frac{\partial\psi}{\partial t}+f(x,t)\frac{\partial\psi}{\partial x}+\frac{1}{2}\sum_{i=1}^{n}g_{i}^{2}(x,t)\frac{\partial^{2}\psi}{\partial x^{2}}\right)dt+\frac{\partial\psi}{\partial x}\sum_{i=1}^{n}g_{i}(x,t)dW_{i}
```

Notice that this is the same as the normal chain rule formula, but now there is a seconf order correction to the main variable. The soltuion is given by the integral form:

```math
x(t)=x(0)+\int_{0}^{t}f(x(s))ds+\int_{0}^{t}\sum_{i=1}^{m}g_{i}(x(s))dW_{i}.
```

Note that we can also generalize Ito's lemma to the multidimensional
```math
\mathbf{X}\in\mathbb{R}^{n}
```

case:

```math
d\psi(\mathbf{X})=\left<\frac{\partial\psi}{\partial\mathbf{X}},f(\mathbf{X})\right>dt+\sum_{i=1}^{m}\left<\frac{\partial\psi}{\partial\mathbf{X}},g_{i}(\mathbf{X})\right>dW_{i}+\frac{1}{2}\sum_{i=1}^{m}g_{i}(\mathbf{X})^{T}\nabla^{2}\psi(\mathbf{X})g_{i}(\mathbf{X})dt
```

There are many other rules as well:

- Product Rule: ``d(X_{t}Y_{t})=X_{t}dY+Y_{t}dX+dXdY``.

- Integration By Parts: ``\int_{0}^{t}X_{t}dY_{t}=X_{t}Y_{t}-X_{0}Y_{0}-\int_{0}^{t}Y_{t}dX_{t}-\int_{0}^{t}dX_{t}dY_{t}``.

- Invariance: ``\mathbb{E}\left[\left(W(t)-W(s)\right)^{2}\right]=t-s`` for ``t > s``.

- Expectation: ``\mathbb{E}[W(t_{1})W(t_{2})]=\min(t_{1},t_{2})``.

- Independent Increments: ``\mathbb{E}\left[\left(W_{t_{i}}-W_{s_{1}}\right)\left(W_{t_{2}}-W_{s_{2}}\right)\right]=0`` if ``\left[t_{1},s_{1}\right]`` does not overlap ``\left[t_{2},s_{2}\right]``.

- Expectation of the Integral: ``\mathbb{E}\left[\int_{0}^{t}h(t)dW_{t}\right]=\mathbb{E}\left[h(t)dW_{t}\right]=0``.

- Ito Isometry: ``\mathbb{E}\left[\left(\int_{0}^{T}X_{t}dW_{t}\right)\right]=\mathbb{E}\left[\int_{0}^{T}X_{t}^{2}dt\right]``.
