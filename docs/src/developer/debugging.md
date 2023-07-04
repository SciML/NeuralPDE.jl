# Debugging PINN Solutions

#### Note this is all not current right now!

Let's walk through debugging functions for the physics-informed neural network
PDE solvers.

```julia
using NeuralPDE, ModelingToolkit, Flux, Zygote
import ModelingToolkit: Interval, infimum, supremum
# 2d wave equation, neumann boundary condition
@parameters x, t
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)
#2D PDE
C = 1
eq = Dtt(u(x, t)) ~ C^2 * Dxx(u(x, t))

# Initial and boundary conditions
bcs = [u(0, t) ~ 0.0,
    u(1, t) ~ 0.0,
    u(x, 0) ~ x * (1.0 - x),
    Dt(u(x, 0)) ~ 0.0]

# Space and time domains
domains = [x ∈ Interval(0.0, 1.0),
    t ∈ Interval(0.0, 1.0)]

# Neural network
chain = FastChain(FastDense(2, 16, Flux.σ), FastDense(16, 16, Flux.σ), FastDense(16, 1))
init_params = DiffEqFlux.initial_params(chain)

eltypeθ = eltype(init_params)
phi = NeuralPDE.get_phi(chain)
derivative = NeuralPDE.get_numeric_derivative()

u_ = (cord, θ, phi) -> sum(phi(cord, θ))

phi([1, 2], init_params)

phi_ = (p) -> phi(p, init_params)[1]
dphi = Zygote.gradient(phi_, [1.0, 2.0])

dphi1 = derivative(phi, u_, [1.0, 2.0], [[0.0049215667, 0.0]], 1, init_params)
dphi2 = derivative(phi, u_, [1.0, 2.0], [[0.0, 0.0049215667]], 1, init_params)
isapprox(dphi[1][1], dphi1, atol = 1e-8)
isapprox(dphi[1][2], dphi2, atol = 1e-8)

indvars = [x, t]
depvars = [u(x, t)]
dict_depvars_input = Dict(:u => [:x, :t])
dim = length(domains)
dx = 0.1
multioutput = chain isa AbstractArray
strategy = NeuralPDE.GridTraining(dx)
integral = NeuralPDE.get_numeric_integral(strategy, indvars, multioutput, chain, derivative)

_pde_loss_function = NeuralPDE.build_loss_function(eq, indvars, depvars, phi, derivative,
                                                   integral, multioutput, init_params,
                                                   strategy)
```

```
julia> expr_pde_loss_function = NeuralPDE.build_symbolic_loss_function(eq,indvars,depvars,dict_depvars_input,phi,derivative,integral,multioutput,init_params,strategy)

:((cord, var"##θ#529", phi, derivative, integral, u)->begin
          begin
              let (x, t) = (cord[[1], :], cord[[2], :])
                  derivative.(phi, u, cord, Array{Float32,1}[[0.0, 0.0049215667], [0.0, 0.0049215667]], 2, var"##θ#529") .- derivative.(phi, u, cord, Array{Float32,1}[[0.0049215667, 0.0], [0.0049215667, 0.0]], 2, var"##θ#529")
              end
          end
      end)

julia> bc_indvars = NeuralPDE.get_variables(bcs,indvars,depvars)
4-element Array{Array{Any,1},1}:
 [:t]
 [:t]
 [:x]
 [:x]
```

```julia
_bc_loss_functions = [NeuralPDE.build_loss_function(bc, indvars, depvars,
                                                    phi, derivative, integral, multioutput,
                                                    init_params, strategy,
                                                    bc_indvars = bc_indvar)
                      for (bc, bc_indvar) in zip(bcs, bc_indvars)]
```

```
julia> expr_bc_loss_functions = [NeuralPDE.build_symbolic_loss_function(bc,indvars,depvars,dict_depvars_input,
                                                                        phi,derivative,integral,multioutput,init_params,strategy,
                                                                        bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]
4-element Array{Expr,1}:
 :((cord, var"##θ#529", phi, derivative, integral, u)->begin
          begin
              let (x, t) = (cord[[1], :], cord[[2], :])
                  u.(cord, var"##θ#529", phi) .- 0.0
              end
          end
      end)
 :((cord, var"##θ#529", phi, derivative, integral, u)->begin
          begin
              let (x, t) = (cord[[1], :], cord[[2], :])
                  u.(cord, var"##θ#529", phi) .- 0.0
              end
          end
      end)
 :((cord, var"##θ#529", phi, derivative, integral, u)->begin
          begin
              let (x, t) = (cord[[1], :], cord[[2], :])
                  u.(cord, var"##θ#529", phi) .- (*).(x, (+).(1.0, (*).(-1, x)))
              end
          end
      end)
 :((cord, var"##θ#529", phi, derivative, integral, u)->begin
          begin
              let (x, t) = (cord[[1], :], cord[[2], :])
                  derivative.(phi, u, cord, Array{Float32,1}[[0.0, 0.0049215667]], 1, var"##θ#529") .- 0.0
              end
          end
      end)
```

```julia
train_sets = NeuralPDE.generate_training_sets(domains, dx, [eq], bcs, eltypeθ, indvars,
                                              depvars)
pde_train_set, bcs_train_set = train_sets
```

```
julia> pde_train_set
1-element Array{Array{Float32,2},1}:
 [0.1 0.2 … 0.8 0.9; 0.1 0.1 … 1.0 1.0]


julia> bcs_train_set
4-element Array{Array{Float32,2},1}:
 [0.0 0.0 … 0.0 0.0; 0.0 0.1 … 0.9 1.0]
 [1.0 1.0 … 1.0 1.0; 0.0 0.1 … 0.9 1.0]
 [0.0 0.1 … 0.9 1.0; 0.0 0.0 … 0.0 0.0]
 [0.0 0.1 … 0.9 1.0; 0.0 0.0 … 0.0 0.0]
```

```julia
pde_bounds, bcs_bounds = NeuralPDE.get_bounds(domains, [eq], bcs, eltypeθ, indvars, depvars,
                                              NeuralPDE.StochasticTraining(100))
```

```
julia> pde_bounds
1-element Vector{Vector{Any}}:
 [Float32[0.01, 0.99], Float32[0.01, 0.99]]

julia> bcs_bounds
4-element Vector{Vector{Any}}:
 [0, Float32[0.0, 1.0]]
 [1, Float32[0.0, 1.0]]
 [Float32[0.0, 1.0], 0]
 [Float32[0.0, 1.0], 0]
```

```julia
discretization = NeuralPDE.PhysicsInformedNN(chain, strategy)

@named pde_system = PDESystem(eq, bcs, domains, indvars, depvars)
prob = NeuralPDE.discretize(pde_system, discretization)

expr_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
expr_pde_loss_function, expr_bc_loss_functions = expr_prob
```
