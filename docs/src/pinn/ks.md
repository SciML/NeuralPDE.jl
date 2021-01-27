# Kuramoto–Sivashinsky equation

Let's consider the Kuramoto–Sivashinsky equation, which contains a 4th-order derivative:

![KS](https://user-images.githubusercontent.com/12683885/91025423-09fb2b00-e602-11ea-8f5c-61e49e4fb54e.png)

with the initial and boundary conditions:

![bs](https://user-images.githubusercontent.com/12683885/91025570-3fa01400-e602-11ea-8fd7-5b0e250a67a4.png)

with physics-informed neural networks.

```julia
@parameters x, t
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dx2 = Differential(x)^2
Dx3 = Differential(x)^3
Dx4 = Differential(x)^4

α = 1
β = 4
γ = 1
eq = Dt(u(x,t)) + u(x,t)*Dx(u(x,t)) + α*Dx2(u(x,t)) + β*Dx3(u(x,t)) + γ*Dx4(u(x,t)) ~ 0

u_analytic(x,t;z = -x/2+t) = 11 + 15*tanh(z) -15*tanh(z)^2 - 15*tanh(z)^3
du(x,t;z = -x/2+t) = 15/2*(tanh(z) + 1)*(3*tanh(z) - 1)*sech(z)^2

bcs = [u(x,0) ~ u_analytic(x,0),
       u(-10,t) ~ u_analytic(-10,t),
       u(10,t) ~ u_analytic(10,t),
       Dx(u(-10,t)) ~ du(-10,t),
       Dx(u(10,t)) ~ du(10,t)]

# Space and time domains
domains = [x ∈ IntervalDomain(-10.0,10.0),
           t ∈ IntervalDomain(0.0,1.0)]
# Discretization
dx = 0.4; dt = 0.2

# Neural network
chain = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))

discretization = PhysicsInformedNN(chain, GridTraining([dx,dt]))
pde_system = PDESystem(eq,bcs,domains,[x,t],[u])
prob = discretize(pde_system,discretization)

opt = Optim.BFGS()
res = GalacticOptim.solve(prob,opt; cb = cb, maxiters=2000)
phi = discretization.phi
```

And some analysis:

```julia
xs,ts = [domain.domain.lower:dx:domain.domain.upper for (domain,dx) in zip(domains,[dx/10,dt])]

u_predict = [[first(phi([x,t],res.minimizer)) for x in xs] for t in ts]
u_real = [[u_analytic(x,t) for x in xs] for t in ts]
diff_u = [[abs(u_analytic(x,t) -first(phi([x,t],res.minimizer)))  for x in xs] for t in ts]

p1 =plot(xs,u_predict,title = "predict")
p2 =plot(xs,u_real,title = "analytic")
p3 =plot(xs,diff_u,title = "error")
plot(p1,p2,p3)
```

![plotks](https://user-images.githubusercontent.com/12683885/91025889-a6253200-e602-11ea-8f61-8e6e2488e025.png)
