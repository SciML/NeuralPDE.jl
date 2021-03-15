# 1-D Burgers' Equation With Low-Level API

Let's consider the Burgers’ equation:

![burgers](https://user-images.githubusercontent.com/12683885/90985032-b6e19380-e581-11ea-89ee-cdfdc4ecf075.png)

with Physics-Informed Neural Networks. Here is an example of using the low-level API:

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux

@parameters t, x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

#2D PDE
eq  = Dt(u(t,x)) + u(t,x)*Dx(u(t,x)) - (0.01/pi)*Dxx(u(t,x)) ~ 0

# Initial and boundary conditions
bcs = [u(0,x) ~ -sin(pi*x),
       u(t,-1) ~ 0.,
       u(t,1) ~ 0.]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,1.0),
           x ∈ IntervalDomain(-1.0,1.0)]
# Discretization
dx = 0.1
# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

strategy = GridTraining(dx)

phi = NeuralPDE.get_phi(chain)
derivative = NeuralPDE.get_numeric_derivative()
initθ = DiffEqFlux.initial_params(chain)

indvars = [t,x]
depvars = [u]

_pde_loss_function = NeuralPDE.build_loss_function(eq,indvars,depvars,
                                         phi,derivative,initθ,strategy)

bc_indvars = NeuralPDE.get_bc_varibles(bcs,indvars,depvars)
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,
                                          phi,derivative,initθ,strategy,
                                          bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

train_sets = NeuralPDE.generate_training_sets(domains,dx,[eq],bcs,indvars,depvars)
train_domain_set, train_bound_set = train_sets


pde_loss_function = NeuralPDE.get_loss_function(_pde_loss_function,
                                      train_domain_set,
                                      strategy)

bc_loss_function = NeuralPDE.get_loss_function(_bc_loss_functions,
                                     train_bound_set,
                                     strategy)

function loss_function_(θ,p)
    return pde_loss_function(θ) + bc_loss_function(θ)
end
f = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f, initθ)

cb = function (p,l)
    println("Current losses are: ", pde_loss_function(p), " , ",  bc_loss_function(p))
    return false
end

# optimizer
opt = Optim.BFGS()
res = GalacticOptim.solve(prob, opt; cb = cb, maxiters=2000)
```

And some analysis:

```julia
using Plots

ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
u_predict_contourf = reshape([first(phi([t,x],res.minimizer)) for t in ts for x in xs] ,length(xs),length(ts))
plot(ts, xs, u_predict_contourf, linetype=:contourf,title = "predict")

u_predict = [[first(phi([t,x],res.minimizer)) for x in xs] for t in ts ]
p1= plot(xs, u_predict[2],title = "t = 0.1");
p2= plot(xs, u_predict[6],title = "t = 0.5");
p3= plot(xs, u_predict[end],title = "t = 1");
plot(p1,p2,p3)
```

![burgers](https://user-images.githubusercontent.com/12683885/90984874-a0870800-e580-11ea-9fd4-af8a4e3c523e.png)


![burgers2](https://user-images.githubusercontent.com/12683885/90984856-8c430b00-e580-11ea-9206-1a88ebd24ca0.png)

[See low-level API](@ref Physics-Informed Neural Networks)
