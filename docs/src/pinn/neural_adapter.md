## Neural Adapter

![image](https://user-images.githubusercontent.com/12683885/127149639-c2a8066f-9a25-4889-b313-5d4403567300.png)

2D Poisson equation with Neural adapter

```julia
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

# Initial and boundary conditions
bcs = [u(0,y) ~ 0.0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
       u(x,0) ~ 0.0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]
# Space and time domains
domains = [x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0)]
quadrature_strategy = NeuralPDE.QuadratureTraining(reltol=1e-2,abstol=1e-2,
                                                   maxiters =50, batch=100)
inner = 8
af = Flux.tanh
chain1 = Chain(Dense(2,inner,af),
               Dense(inner,inner,af),
               Dense(inner,1))

initθ = Float64.(DiffEqFlux.initial_params(chain1))
discretization = NeuralPDE.PhysicsInformedNN(chain1,
                                             quadrature_strategy;
                                             init_params = initθ)

pde_system = PDESystem(eq,bcs,domains,[x,y],[u])
prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, BFGS();  maxiters=2000)
phi = discretization.phi

inner_ = 8
af = Flux.tanh
chain2 = FastChain(FastDense(2,inner_,af),
                   FastDense(inner_,inner_,af),
                   FastDense(inner_,inner_,af),
                   FastDense(inner_,1))

initθ2 =Float64.(DiffEqFlux.initial_params(chain2))

function loss(cord,θ)
    chain2(cord,θ) .- phi(cord,res.minimizer)
end

quadrature_strategy = NeuralPDE.QuadratureTraining(reltol=1e-2,abstol=1e-2,
                                                   maxiters =50, batch=100)


prob_ = NeuralPDE.neural_adapter(loss, initθ2, pde_system, quadrature_strategy)
res_ = GalacticOptim.solve(prob_, BFGS(); maxiters=1000)

parameterless_type_θ = DiffEqBase.parameterless_type(initθ2)
phi_ = NeuralPDE.get_phi(chain2,parameterless_type_θ)

xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_predict_ =  reshape([first(phi_([x,y],res_.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))

using Plots
diff_u = u_predict .- u_real
diff_u_ = u_predicts .- u_real
p1 = plot(xs, ys, u_predict, linetype=:contourf,title = "first predict");
p2 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
p4 = plot(xs, ys, u_predicts[i],linetype=:contourf,title = "second predict");
p5 = plot(xs, ys, diff_u_,linetype=:contourf,title = "error_");
plot(p1,p2,p3,p4,p5)

```

## Domain decomposition

![domain_decomposition](https://user-images.githubusercontent.com/12683885/127149752-a4ecea50-2984-45d8-b0d4-d2eadecf58e7.png)

```julia
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

eq  = Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)

bcs = [u(0,y) ~ 0.0, u(1,y) ~ -sin(pi*1)*sin(pi*y),
       u(x,0) ~ 0.0, u(x,1) ~ -sin(pi*x)*sin(pi*1)]

# Space
x_0 = 0.0
x_end = 1.0
x_domain = Interval(x_0, x_end)
y_domain = Interval(0.0, 1.0)

count_decomp = 10

# Neural network
af = Flux.tanh
inner = 10
chains = [FastChain(FastDense(2, inner, af), FastDense(inner, inner, af), FastDense(inner, 1)) for _ in 1:count_decomp]
initθs = map(c -> Float64.(c), DiffEqFlux.initial_params.(chains))

xs_ = infimum(x_domain):1/count_decomp:supremum(x_domain)
xs_domain = [(xs_[i], xs_[i+1]) for i in 1:length(xs_)-1]
domains_map = map(xs_domain) do (xs_dom)
    x_domain_ = Interval(xs_dom...)
    domains_ = [x ∈ x_domain_,
                y ∈ y_domain]
end

analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)
function create_bcs(bcs,x_domain_,phi_bound)
    x_0, x_e =  x_domain_.left, x_domain_.right
    if x_0 == 0.0
        bcs = [u(0,y) ~ 0.0,
               u(x_e,y) ~ analytic_sol_func(x_e,y),
               u(x,0) ~ 0.0,
               u(x,1) ~ -sin(pi*x)*sin(pi*1)]
        return bcs
    end
    bcs = [u(x_0,y) ~ phi_bound(x_0,y),
           u(x_e,y) ~ analytic_sol_func(x_e,y),
           u(x,0) ~ 0.0,
           u(x,1) ~ -sin(pi*x)*sin(pi*1)]
    bcs
end

reses = []
phis = []
pde_system_map = []

for i in 1:count_decomp
    println("decomposition $i")
    domains_ = domains_map[i]
    phi_in(cord) = phis[i-1](cord,reses[i-1].minimizer)
    phi_bound(x,y) = phi_in(vcat(x,y))
    @register phi_bound(x,y)
    Base.Broadcast.broadcasted(::typeof(phi_bound), x,y) = phi_bound(x,y)
    bcs_ = create_bcs(bcs,domains_[1].domain, phi_bound)
    pde_system_ = PDESystem(eq, bcs_, domains_, [x, y], [u])
    push!(pde_system_map,pde_system_)
    strategy = NeuralPDE.GridTraining([0.1/count_decomp, 0.1])

    discretization = NeuralPDE.PhysicsInformedNN(chains[i], strategy; init_params=initθs[i])

    prob = NeuralPDE.discretize(pde_system_,discretization)
    symprob = NeuralPDE.symbolic_discretize(pde_system_,discretization)
    res_ = GalacticOptim.solve(prob, BFGS(), maxiters=1000)
    phi = discretization.phi
    push!(reses, res_)
    push!(phis, phi)
end

using Plots
function plot_(i)
    xs, ys = [infimum(d.domain):dx:supremum(d.domain) for (dx,d) in zip([0.001,0.01], domains_map[i])]
    u_predict = reshape([first(phis[i]([x,y],reses[i].minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
    u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
    diff_u = abs.(u_predict .- u_real)
    p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
    p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
    p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
    plot(p1,p2,p3)
end
ps =[plot_(i) for i in 1:10]

inner_ = 18
af = Flux.tanh
chain2 = FastChain(FastDense(2,inner_,af),
                   FastDense(inner_,inner_,af),
                   FastDense(inner_,inner_,af),
                   FastDense(inner_,inner_,af),
                   FastDense(inner_,1))

initθ2 =Float64.(DiffEqFlux.initial_params(chain2))

domains = [x ∈ x_domain,
           y ∈ y_domain]
pde_system = PDESystem(eq, bcs, domains, [x, y], [u])
symprob = NeuralPDE.symbolic_discretize(pde_system,discretization)

losses = map(1:count_decomp) do i
    loss(cord,θ) = chain2(cord,θ) .- phis[i](cord,reses[i].minimizer)
end

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

prob_ = NeuralPDE.neural_adapter(losses,initθ2, pde_system_map,NeuralPDE.GridTraining([0.1/count_decomp,0.1]))
res_ = GalacticOptim.solve(prob_, BFGS();cb=cb, maxiters=2000)
prob_ = NeuralPDE.neural_adapter(losses,res_.minimizer, pde_system_map, NeuralPDE.GridTraining([0.05/count_decomp,0.05]))
res_ = GalacticOptim.solve(prob_, BFGS();cb=cb,  maxiters=1000)

parameterless_type_θ = DiffEqBase.parameterless_type(initθ2)
phi_ = NeuralPDE.get_phi(chain2,parameterless_type_θ)

xs,ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
u_predict_ = reshape([first(phi_([x,y],res_.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = u_predict_ .- u_real

p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
p2 = plot(xs, ys, u_predict_, linetype=:contourf,title = "predict");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)
```

![decomp](https://user-images.githubusercontent.com/12683885/127161424-46222366-8831-4f94-b44a-989dd22b0eb0.png)
