using ModelingToolkit
using Flux
using NeuralPDE
using GalacticOptim
using DiffEqFlux
using DomainSets
import ModelingToolkit: Interval, infimum, supremum

@parameters t x v
@variables f(..) E(..)
Dx = Differential(x)
Dt = Differential(t)
Dv = Differential(v)

# Constants
μ_0 = 1.25663706212e-6 # N A⁻²
ε_0 = 8.8541878128e-12 # F ms⁻¹
e = 1.602176634e-19 # Coulombs
m_e = 9.10938188e-31 # Kg
n_0 = 1
v_th = sqrt(2)

1 / (v_th * sqrt(2 * pi)) * exp(-v^2 / (2 * v_th^2))

# Integrals

Iv = Integral((v,x) in DomainSets.ProductDomain(ClosedInterval(-Inf ,Inf), ClosedInterval(-Inf ,Inf)))

eqs = [Dt(f(t, x, v)) ~ -v * Dx(f(t, x, v)) - e / m_e * E(t, x) * Dv(f(t, x, v))]

bcs = [f(0, x, v) ~ 1 / (v_th * sqrt(2π)) * exp(-v^2 / (2 * v_th^2)),
    E(0, x) ~ e * n_0 / ε_0 * (Iv(f(0, x, v) * v) - 1)]

domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0),
    v ∈ Interval(0.0, 1.0)]

# Neural Network
chain = [FastChain(FastDense(3, 16, Flux.σ), FastDense(16, 16, Flux.σ), FastDense(16, 1)),
    FastChain(FastDense(2, 16, Flux.σ), FastDense(16, 16, Flux.σ), FastDense(16, 1))]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

discretization = NeuralPDE.PhysicsInformedNN(chain, QuadratureTraining(), init_params = initθ)
@named pde_system = PDESystem(eqs, bcs, domains, [t, x, v], [f(t, x, v), E(t, x)])
prob = SciMLBase.symbolic_discretize(pde_system, discretization)
prob = SciMLBase.discretize(pde_system, discretization)


### MWE

@parameters x
@variables u(..)
chain = FastChain(FastDense(1,15,Flux.σ),FastDense(15,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
strategy = NeuralPDE.GridTraining(0.1)
indvars = [x]
depvars = [u(x)]
parameterless_type_θ = Float64
phi = NeuralPDE.get_phi(chain,parameterless_type_θ)

u_ = (cord, θ, phi)-> phi(cord, θ)
cord = [1.]
lb, ub = [-1.], [1.]
θ = initθ

derivative = NeuralPDE.get_numeric_derivative()
integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative)

transform(τ) = τ ./ (1 .- τ.^2)
integral_f = (cord, var"##θ#332", phi, derivative, integral, u_, p)->begin
          begin
              let (x,) = (cord[[1], :],)
                  begin
                      cord1 = vcat(x)
                  end
                  u_(transform(cord1), var"##θ#332", phi)
              end
          end
      end

integral_f(cord, θ, phi, derivative, nothing, u_, nothing)
inyf_ = (θ) -> sum(abs2,integral_f(cord, θ, phi, derivative, nothing, u_, nothing))
inyf_(θ)

@show Zygote.gradient(inyf_,θ)