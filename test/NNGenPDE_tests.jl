using Test, Flux,Optim#, NeuralNetDiffEq
using DiffEqDevTools
using Random
using DiffEqFlux
using Plots
using Adapt
using ModelingToolkit
using DiffEqOperators, DiffEqBase, LinearAlgebra

#TODO avoid dependence in pde_func from u, du and du2
dim =2
dx= 0.1
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

initθ = DiffEqFlux.initial_params(chain)
phi = (x,θ) -> first(chain(adapt(typeof(θ),x),θ))
epsilon(dx) = cbrt(eps(typeof(dx)))
e = epsilon(dx)
eps_masks = [
         [[e]],
         [[e, 0.0], [0.0,e]],
         [[e,0.0,0.0], [0.0,e,0.0],[0.0,0.0,e]]
         ]
u = (x,θ) -> phi(x,θ)
du = (x,θ,n) -> (phi(collect(x+eps_masks[dim][n]),θ) - phi(x,θ))/epsilon(dx)
du2 = (x,θ,n) -> (phi(x+eps_masks[dim][n],θ) - 2phi(x,θ) + phi(x-eps_masks[dim][n],θ))/epsilon(dx)^2



## Example 1  1-dim ode
@parameters t
domains = [t ∈ IntervalDomain(0.0,1.0)]

function pde_func(cord,θ)
    t = cord[1]
    -du(cord,θ,1) + t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u(cord,θ)*(t + ((1+3*(t^2))/(1+t+t^3)))
end

bound_cond_funcs(t) = exp(-(t^2)/2)/(1+t+t^3) + t^2
linear_analytic_func(t) =  exp(-(t^2)/2)/(1+t+t^3) + t^2

discretization = NeuralNetDiffEq.Discretization(0.1)
spaces = NeuralNetDiffEq.Spaces(domains,discretization)
dim = length(domains)

opt = Flux.ADAM(0.1)
chain = FastChain(FastDense(dim,12,Flux.σ),FastDense(12,1))

prob = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,bound_cond_funcs, spaces, dim)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=1000)

ts = [domain.domain.lower:discretization.dxs/10:domain.domain.upper for domain in domains][1]

u_real  = [linear_analytic_func(t) for t in ts]
u_predict  = [first(phi(t,res.minimizer)) for t in ts]

t_plot = collect(ts)
plot(t_plot ,u_real)
plot!(t_plot ,u_predict)

## Example 2 2d Poisson equation du2/dx2 + du2/dy2 =  1
@parameters x y
# @variables u(..)
# @derivatives Dt'~t
# @derivatives Dxx''~x
domains = [x ∈ IntervalDomain(0.0,1.0),
           y ∈ IntervalDomain(0.0,1.0)]

# eq  = Dtt(u(t,x)) + Dxx(u(t,x)) ~ 1
# bcs = [u(0,x) ~ 0.0f0, u(1,x) ~ 0.0f0, u(t,0) ~ 0.0f0, u(t,1) ~ 0.0f0]

function pde_func(x,θ)
    du2(x,θ,1) + du2(x,θ,2) + 1
end
# TODO make bound_cond_funcs
bound_cond_funcs(x,y) = 0.f0 #u(x,y) = 0 on edge

discretization = NeuralNetDiffEq.Discretization(0.1) #MOLFiniteDifference(0.1)
spaces = NeuralNetDiffEq.Spaces(domains,discretization)
dim = length(domains)

opt = Flux.ADAM(0.1)
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

prob = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,bound_cond_funcs, spaces, dim)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=500)

xs,ys = [domain.domain.lower:discretization.dxs:domain.domain.upper for domain in domains]
u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
plot(xs, ys, u_predict, st=:surface)


## Example 3  3dim Poisson equation du2/dx2 + du2/dy2+ du2/dt2 = -sin(pi*x)*sin(pi*y)*sin(pi*t)
@parameters x y t

domains = [x ∈ IntervalDomain(0.0,1.0),
           y ∈ IntervalDomain(0.0,1.0),
           t ∈ IntervalDomain(0.0,0.5)]

function pde_func(cord,θ)
    x,y,t = cord
    du2(cord,θ,3) + du2(cord,θ,1) + du2(cord,θ,2) + sin(pi*x)*sin(pi*y)*sin(pi*t)
end

bound_cond_funcs(x,y,t) = sin(pi*x)*sin(pi*y)*sin(pi*t)/3pi^2
linear_analytic_func(x,y,t) = sin(pi*x)*sin(pi*y)*sin(pi*t)/3pi^2

discretization = NeuralNetDiffEq.Discretization(0.1)
spaces = NeuralNetDiffEq.Spaces(domains,discretization)
dim = length(domains)

opt = Flux.ADAM(0.1)
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

prob = NeuralNetDiffEq.GeneranNNPDEProblem(pde_func,bound_cond_funcs, spaces, dim)
alg = NeuralNetDiffEq.NNGeneralPDE(chain,opt,autodiff=false)
phi,res  = NeuralNetDiffEq.solve(prob,alg,verbose=true, maxiters=500)

xs,ys,ts = [domain.domain.lower:discretization.dxs:domain.domain.upper for domain in domains]

u_real = [reshape([linear_analytic_func(x,y,t) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]
u_predict = [reshape([first(phi([x,y,t],res.minimizer)) for x in xs  for y in ys], (length(xs),length(ys)))  for t in ts ]
p1 =plot(xs, ys, u_predict[3], st=:surface);
p2 = plot(xs, ys, u_real[3], st=:surface);
plot(p1,p2)
