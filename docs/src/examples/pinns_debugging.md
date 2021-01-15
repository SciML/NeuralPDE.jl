### Debugging

```julia
using NeuralPDE, ModelingToolkit, Flux, DiffEqFlux, Zygote
# 2d wave equation, neumann boundary condition
@parameters x, t
@variables u(..)
@derivatives Dxx''~x
@derivatives Dtt''~t
@derivatives Dt'~t

#2D PDE
C=1
eq  = Dtt(u(x,t)) ~ C^2*Dxx(u(x,t))

# Initial and boundary conditions
bcs = [u(0,t) ~ 0.,
       u(1,t) ~ 0.,
       u(x,0) ~ x*(1. - x),
       Dt(u(x,0)) ~ 0. ]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0),
           t ∈ IntervalDomain(0.0,1.0)]

# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
initθ = DiffEqFlux.initial_params(chain)

phi = get_phi(chain)
derivative = get_numeric_derivative()

u_ = (cord, θ, phi)->sum(phi(cord, θ))

phi([1,2], initθ)

phi_ = (p) -> phi(p, initθ)[1]
dphi = Zygote.gradient(phi_,[1.,2.])

dphi1 = derivative(phi,u_,[1.,2.],[[ 0.0049215667, 0.0]],1,initθ)
dphi2 = derivative(phi,u_,[1.,2.],[[0.0,  0.0049215667]],1,initθ)
isapprox(dphi[1][1], dphi1, atol=1e-8)
isapprox(dphi[1][2], dphi2, atol=1e-8)


indvars = [x,t]
depvars = [u]
dim = length(domains)

_pde_loss_function = build_loss_function(eq,indvars,depvars,phi, derivative,initθ)

julia> expr_pde_loss_function = build_symbolic_loss_function(eq,indvars,depvars,phi,derivative,initθ)

:((cord, var"##θ#1211", phi, derivative, u)->begin
          begin
              let (x, t) = (cord[1], cord[2])
                  [derivative(phi, u, [x, t], Array{Float32,1}[[0.0, 0.0049215667], [0.0, 0.0049215667]], 2, var"##θ#1211") - derivative(phi, u, [x, t], Array{Float32,1}[[0.0049215667, 0.0], [0.0049215667, 0.0]], 2, var"##θ#1211")]
              end
          end

julia> bc_indvars = get_bc_varibles(bcs,indvars,depvars)
4-element Array{Array{Any,1},1}:
 [:t]
 [:t]
 [:x]
 [:x]

_bc_loss_functions = [build_loss_function(bc,indvars,depvars,
                                                    phi, derivative,initθ,
                                                    bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

julia> expr_bc_loss_functions = [build_symbolic_loss_function(bc,indvars,depvars,
                                                                 phi, derivative,initθ,
                                                                 bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]                                         
4-element Array{Expr,1}:
 :((cord, var"##θ#1211", phi, derivative, u)->begin
          begin
              let (t,) = (cord[1],)
                  [u([0, t], var"##θ#1211", phi) - 0.0]
              end
          end
      end)
 :((cord, var"##θ#1211", phi, derivative, u)->begin
          begin
              let (t,) = (cord[1],)
                  [u([1, t], var"##θ#1211", phi) - 0.0]
              end
          end
      end)
 :((cord, var"##θ#1211", phi, derivative, u)->begin
          begin
              let (x,) = (cord[1],)
                  [u([x, 0], var"##θ#1211", phi) - (*)(x, (-)(1.0, x))]
              end
          end
      end)
 :((cord, var"##θ#1211", phi, derivative, u)->begin
          begin
              let (x,) = (cord[1],)
                  [derivative(phi, u, [x, 0], Array{Float32,1}[[0.0, 0.0049215667]], 1, var"##θ#1211") - 0.0]
              end
          end
      end)
dx=0.1
train_sets = generate_training_sets(domains,dx,bcs,indvars,depvars)
pde_train_set,bcs_train_set,train_set = train_sets

julia> pde_train_set
90-element Array{Array{Float64,1},1}:
 [0.1, 0.1]
 [0.2, 0.1]
 [0.3, 0.1]
 ⋮
 [0.7, 1.0]
 [0.8, 1.0]
 [0.9, 1.0]

 julia> bcs_train_set
 4-element Array{Array{Array{Float64,1},1},1}:
  [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]]
  [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]]
  [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]]
  [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]]

pde_bounds, bcs_bounds = get_bounds(domains,bcs,indvars,depvars)

julia> pde_bounds
2-element Array{Array{Float64,1},1}:
 [0.0, 0.0]
 [1.0, 1.0]

julia> bcs_bounds
2-element Array{Array{Array{Float64,1},1},1}:
 [[0.0], [0.0], [0.0], [0.0]]
 [[1.0], [1.0], [1.0], [1.0]]

strategy = GridTraining(dx)
discretization = PhysicsInformedNN(chain,strategy)

pde_system = PDESystem(eq,bcs,domains,indvars,depvars)
prob = discretize(pde_system,discretization)

expr_prob = symbolic_discretize(pde_system,discretization)
expr_pde_loss_function , expr_bc_loss_functions = expr_prob

```
