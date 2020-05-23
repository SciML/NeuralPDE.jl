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