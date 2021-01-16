# Fokker-Planck Equation

Let's consider the Fokker-Planck equation:

![fke](https://user-images.githubusercontent.com/12683885/91547965-58c00200-e92d-11ea-8d7b-f20ba79ed7c1.png)

which must satisfy the normalization condition:

![nc](https://user-images.githubusercontent.com/12683885/91548028-74c3a380-e92d-11ea-8ee4-ac2a1c780808.png)

with the boundary conditions:

![bc](https://user-images.githubusercontent.com/12683885/91548102-902eae80-e92d-11ea-8956-736a54e9591e.png)


```julia
# the example is taken from this article https://arxiv.org/abs/1910.10503
@parameters x
@variables p(..)
@derivatives Dx'~x
@derivatives Dxx''~x

#2D PDE
α = 0.3
β = 0.5
_σ = 0.5
# Discretization
dx = 0.05
# here we use normalization condition: dx*p(x) ~ 1 in order to get a non-zero solution.
eq  = [(α - 3*β*x^2)*p(x) + (α*x - β*x^3)*Dx(p(x)) ~ (_σ^2/2)*Dxx(p(x)),
       dx*p(x) ~ 1.]

# Initial and boundary conditions
bcs = [p(-2.2) ~ 0. ,p(2.2) ~ 0. , p(-2.2) ~ p(2.2)]

# Space and time domains
domains = [x ∈ IntervalDomain(-2.2,2.2)]

# Neural network
chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))

discretization = PhysicsInformedNN(chain,GridTraining(dx))

pde_system = PDESystem(eq,bcs,domains,[x],[p])
prob = discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, BFGS(); cb = cb, maxiters=8000)
phi = discretization.phi
```

And some analysis:

```julia
analytic_sol_func(x) = 28.022*exp((1/(2*_σ^2))*(2*α*x^2 - β*x^4))

xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

plot(xs ,u_real, label = "analytic")
plot!(xs ,u_predict, label = "predict")
```

![fkplot](https://user-images.githubusercontent.com/12683885/91551079-47c5bf80-e932-11ea-906e-23c84e2846dc.png)
