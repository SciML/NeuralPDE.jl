# Systems of PDEs

In this example, we will solve the PDE system:

![pdesystem](https://user-images.githubusercontent.com/12683885/90978370-22157080-e556-11ea-92b3-d65cb9aa3115.png)

with the initial conditions:

![Initial](https://user-images.githubusercontent.com/12683885/90978670-322e4f80-e558-11ea-8157-a0b6ec84e121.png)

and the boundary conditions:

![boundary](https://user-images.githubusercontent.com/12683885/90978689-4c682d80-e558-11ea-8e51-080bd02a1856.png)

with physics-informed neural networks.

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature
import ModelingToolkit: Interval, infimum, supremum

@parameters t, x
@variables u1(..), u2(..), u3(..)
Dt = Differential(t)
Dtt = Differential(t)^2
Dx = Differential(x)
Dxx = Differential(x)^2

eqs = [Dtt(u1(t,x)) ~ Dxx(u1(t,x)) + u3(t,x)*sin(pi*x),
       Dtt(u2(t,x)) ~ Dxx(u2(t,x)) + u3(t,x)*cos(pi*x),
       0. ~ u1(t,x)*sin(pi*x) + u2(t,x)*cos(pi*x) - exp(-t)]

bcs = [u1(0,x) ~ sin(pi*x),
       u2(0,x) ~ cos(pi*x),
       Dt(u1(0,x)) ~ -sin(pi*x),
       Dt(u2(0,x)) ~ -cos(pi*x),
       u1(t,0) ~ 0.,
       u2(t,0) ~ exp(-t),
       u1(t,1) ~ 0.,
       u2(t,1) ~ -exp(-t)]


# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0)]

# Neural network
input_ = length(domains)
n = 15
chain =[FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1)) for _ in 1:3]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

_strategy = NeuralPDE.QuadratureTraining()
discretization = NeuralPDE.PhysicsInformedNN(chain, _strategy, init_params= initθ)

pde_system = PDESystem(eqs,bcs,domains,[t,x],[u1,u2,u3])
prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

pde_inner_loss_functions = prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents
bcs_inner_loss_functions = prob.f.f.loss_function.bcs_loss_function.bc_loss_functions.contents

cb = function (p,l)
    println("loss: ", l )
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return false
end

res = GalacticOptim.solve(prob,BFGS(); cb = cb, maxiters=5000)

phi = discretization.phi
```


Low-level api


```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature
import ModelingToolkit: Interval, infimum, supremum

@parameters t, x
@variables u1(..), u2(..), u3(..)
Dt = Differential(t)
Dtt = Differential(t)^2
Dx = Differential(x)
Dxx = Differential(x)^2

eqs = [Dtt(u1(t,x)) ~ Dxx(u1(t,x)) + u3(t,x)*sin(pi*x),
       Dtt(u2(t,x)) ~ Dxx(u2(t,x)) + u3(t,x)*cos(pi*x),
       0. ~ u1(t,x)*sin(pi*x) + u2(t,x)*cos(pi*x) - exp(-t)]

bcs = [u1(0,x) ~ sin(pi*x),
       u2(0,x) ~ cos(pi*x),
       Dt(u1(0,x)) ~ -sin(pi*x),
       Dt(u2(0,x)) ~ -cos(pi*x),
       u1(t,0) ~ 0.,
       u2(t,0) ~ exp(-t),
       u1(t,1) ~ 0.,
       u2(t,1) ~ -exp(-t)]

# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0)]

# Neural network
input_ = length(domains)
n = 15
chain =[FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1)) for _ in 1:3]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))
flat_initθ = reduce(vcat,initθ )

eltypeθ = eltype(initθ[1])
parameterless_type_θ = DiffEqBase.parameterless_type(initθ[1])
phi = NeuralPDE.get_phi.(chain,parameterless_type_θ)

map(phi_ -> phi_(rand(2,10), flat_initθ),phi)

derivative = NeuralPDE.get_numeric_derivative()


indvars = [t,x]
depvars = [u1,u2,u3]
dim = length(domains)
quadrature_strategy = NeuralPDE.QuadratureTraining()


_pde_loss_functions = [NeuralPDE.build_loss_function(eq,indvars,depvars,phi,derivative,
                                                     chain,initθ,quadrature_strategy) for eq in  eqs]

map(loss_f -> loss_f(rand(2,10), flat_initθ),_pde_loss_functions)

bc_indvars = NeuralPDE.get_argument(bcs,indvars,depvars)
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars, phi, derivative,
                                                    chain,initθ,quadrature_strategy,
                                                    bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]
map(loss_f -> loss_f(rand(1,10), flat_initθ),_bc_loss_functions)

# dx = 0.1
# train_sets = NeuralPDE.generate_training_sets(domains,dx,eqs,bcs,eltypeθ,indvars,depvars)
# pde_train_set,bcs_train_set = train_sets
pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains,eqs,bcs,eltypeθ,indvars,depvars,quadrature_strategy)

plbs,pubs = pde_bounds
pde_loss_functions = [NeuralPDE.get_loss_function(_loss,
                                                 lb,ub,
                                                 eltypeθ, parameterless_type_θ,
                                                 quadrature_strategy)
                                                 for (_loss,lb,ub) in zip(_pde_loss_functions, plbs,pubs)]

map(l->l(flat_initθ) ,pde_loss_functions)

blbs,bubs = bcs_bounds
bc_loss_functions = [NeuralPDE.get_loss_function(_loss,lb,ub,
                                                 eltypeθ, parameterless_type_θ,
                                                 quadrature_strategy)
                                                 for (_loss,lb,ub) in zip(_bc_loss_functions, blbs,bubs)]

map(l->l(falt_initθ) ,bc_loss_functions)

loss_functions =  [pde_loss_functions;bc_loss_functions]

function loss_function(θ,p)
    sum(map(l->l(θ) ,loss_functions))
end

f_ = OptimizationFunction(loss_function, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f_, flat_initθ)

cb_ = function (p,l)
    println("loss: ", l )
    println("pde losses: ", map(l -> l(p), loss_functions[1:3]))
    println("bcs losses: ", map(l -> l(p), loss_functions[4:end]))
    return false
end

res = GalacticOptim.solve(prob,Optim.BFGS(); cb = cb_, maxiters=5000)
```


And some analysis for both low and high level api:


```julia
using Plots

ts,xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers_ = [res.minimizer[s] for s in sep]

analytic_sol_func(t,x) = [exp(-t)*sin(pi*x), exp(-t)*cos(pi*x), (1+pi^2)*exp(-t)]
u_real  = [[analytic_sol_func(t,x)[i] for t in ts for x in xs] for i in 1:3]
u_predict  = [[phi[i]([t,x],minimizers_[i])[1] for t in ts  for x in xs] for i in 1:3]
diff_u = [abs.(u_real[i] .- u_predict[i] ) for i in 1:3]
for i in 1:3
    p1 = plot(ts, xs, u_real[i],linetype=:contourf,title = "u$i, analytic");
    p2 = plot(ts, xs, u_predict[i],linetype=:contourf,title = "predict");
    p3 = plot(ts, xs, diff_u[i],linetype=:contourf,title = "error");
    plot(p1,p2,p3)
    savefig("sol_u$i")
end
```

![sol_uq1](https://user-images.githubusercontent.com/12683885/122979254-03634e80-d3a0-11eb-985b-d3bae2dddfde.png)

![sol_uq2](https://user-images.githubusercontent.com/12683885/122979278-09592f80-d3a0-11eb-8fee-de3652f138d8.png)

![sol_uq3](https://user-images.githubusercontent.com/12683885/122979288-0e1de380-d3a0-11eb-9005-bfb501959b83.png)


## Derivative neural network approximation

The accuracy and stability of numerical derivative decreases with each successive order.
The accuracy of the entire solution is determined by the worst accuracy of one of the variables, in our case - the highest degree of the derivative.
Derivative neural network approximation is such an approach that using lower-order numeric derivatives and estimates higher-order derivatives with a neural network so that allows an increase in the marginal precision for all optimization.

Since `u3` is only in the first and second equations, that its accuracy during training is determined by the accuracy of the second numerical derivative `u3(t,x)  ~  (Dtt(u1(t,x)) -Dxx(u1(t,x))) / sin(pi*x)`.

We approximate the derivative of the neural network with another neural network `Dt(u1(t,x)) ~ Dtu1(t,x)` and train it along with other equations, and thus we avoid using the second numeric derivative `Dt(Dtu1(t,x))`.



```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature
import ModelingToolkit: Interval, infimum, supremum

@parameters t, x
Dt = Differential(t)
Dx = Differential(x)
@variables u1(..), u2(..), u3(..)
@variables Dxu1(..),Dtu1(..),Dxu2(..),Dtu2(..)

eqs_ = [Dt(Dtu1(t,x)) ~ Dx(Dxu1(t,x)) + u3(t,x)*sin(pi*x),
        Dt(Dtu2(t,x)) ~ Dx(Dxu2(t,x)) + u3(t,x)*cos(pi*x),
        exp(-t) ~ u1(t,x)*sin(pi*x) + u2(t,x)*cos(pi*x)]

bcs_ = [u1(0.,x) ~ sin(pi*x),
       u2(0.,x) ~ cos(pi*x),
       Dt(u1(0,x)) ~ -sin(pi*x),
       Dt(u2(0,x)) ~ -cos(pi*x),
       #Dtu1(0,x) ~ -sin(pi*x),
      # Dtu2(0,x) ~ -cos(pi*x),
       u1(t,0.) ~ 0.,
       u2(t,0.) ~ exp(-t),
       u1(t,1.) ~ 0.,
       u2(t,1.) ~ -exp(-t)]

der_ = [Dt(u1(t,x)) ~ Dtu1(t,x),
        Dt(u2(t,x)) ~ Dtu2(t,x),
        Dx(u1(t,x)) ~ Dxu1(t,x),
        Dx(u2(t,x)) ~ Dxu2(t,x)]

bcs__ = [bcs_;der_]

input_ = length(domains)
n = 15
chain = [FastChain(FastDense(input_,n,Flux.σ),FastDense(n,n,Flux.σ),FastDense(n,1)) for _ in 1:7]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

grid_strategy = NeuralPDE.GridTraining(0.07)
discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             grid_strategy,
                                             init_params= initθ)

vars = [u1,u2,u3,Dxu1,Dtu1,Dxu2,Dtu2]
pde_system = PDESystem(eqs_,bcs__,domains,[t,x],vars)
prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

pde_inner_loss_functions = prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents
inner_loss_functions = prob.f.f.loss_function.bcs_loss_function.bc_loss_functions.contents
bcs_inner_loss_functions = inner_loss_functions[1:7]
aprox_derivative_loss_functions = inner_loss_functions[9:end]

cb = function (p,l)
    println("loss: ", l )
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    println("der_losses: ", map(l_ -> l_(p), aprox_derivative_loss_functions))
    return false
end

res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb, maxiters=2000)
prob = remake(prob,u0=res.minimizer)
res = GalacticOptim.solve(prob,BFGS(); cb = cb, maxiters=10000)

phi = discretization.phi
```


And some analysis:


```julia
using Plots

ts,xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

initθ = discretization.init_params
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers_ = [res.minimizer[s] for s in sep]

u1_real(t,x) = exp(-t)*sin(pi*x)
u2_real(t,x) = exp(-t)*cos(pi*x)
u3_real(t,x) = (1+pi^2)*exp(-t)
Dxu1_real(t,x) = pi*exp(-t)*cos(pi*x)
Dtu1_real(t,x) = -exp(-t)*sin(pi*x)
Dxu2_real(t,x) = -pi*exp(-t)*sin(pi*x)
Dtu2_real(t,x) = -exp(-t)*cos(pi*x)
analytic_sol_func_all(t,x) = [u1_real(t,x), u2_real(t,x), u3_real(t,x),
                              Dxu1_real(t,x),Dtu1_real(t,x),Dxu2_real(t,x),Dtu2_real(t,x)]

u_real  = [[analytic_sol_func_all(t,x)[i] for t in ts for x in xs] for i in 1:7]
u_predict  = [[phi[i]([t,x],minimizers_[i])[1] for t in ts  for x in xs] for i in 1:7]
diff_u = [abs.(u_real[i] .- u_predict[i] ) for i in 1:7]

titles = ["u1","u2","u3","Dtu1","Dtu2","Dxu1","Dxu2"]
for i in 1:7
    p1 = plot(ts, xs, u_real[i], linetype=:contourf,title = "$(titles[i]), analytic");
    p2 = plot(ts, xs, u_predict[i], linetype=:contourf,title = "predict");
    p3 = plot(ts, xs, diff_u[i],linetype=:contourf,title = "error");
    plot(p1,p2,p3)
    savefig("3sol_ub$i")
end
```


![aprNN_sol_u1](https://user-images.githubusercontent.com/12683885/122998551-de79d600-d3b5-11eb-8f5d-59d00178c2ab.png)

![aprNN_sol_u2](https://user-images.githubusercontent.com/12683885/122998567-e3d72080-d3b5-11eb-9024-4072f4b66cda.png)

![aprNN_sol_u3](https://user-images.githubusercontent.com/12683885/122998578-e6d21100-d3b5-11eb-96a5-f64e5593b35e.png)



## Comparison of the second numerical derivative and numerical + neural network derivative


![DDu1](https://user-images.githubusercontent.com/12683885/123113394-3280cb00-d447-11eb-88e3-a8541bbf089f.png)

![DDu2](https://user-images.githubusercontent.com/12683885/123113413-36ace880-d447-11eb-8f6a-4c3caa86e359.png)


## Solving Matrices of PDEs

Also, in addition to systems, we can use the matrix form of PDEs:

```julia
@parameters x y
@variables u[1:2,1:2](..)
@derivatives Dxx''~x
@derivatives Dyy''~y

# matrix PDE
eqs  = @. [(Dxx(u_(x,y)) + Dyy(u_(x,y))) for u_ in u] ~ -sin(pi*x)*sin(pi*y)*[0 1; 0 1]

# Initial and boundary conditions
bcs = [u[1](x,0) ~ x, u[2](x,0) ~ 2, u[3](x,0) ~ 3, u[4](x,0) ~ 4]
```
