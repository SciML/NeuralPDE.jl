
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
Iv = Integral(v in DomainSets.ClosedInterval(-Inf, Inf))

eqs = [Dt(f(t, x, v)) ~ -v * Dx(f(t, x, v)) - e / m_e * E(t, x) * Dv(f(t, x, v))
    Dx(E(t, x)) ~ e * n_0 / ε_0 * (Iv(f(t, x, v)) - 1)]

bcs = [f(0, x, v) ~ 1 / (v_th * sqrt(2π)) * exp(-v^2 / (2 * v_th^2)),
    E(0, x) ~ e * n_0 / ε_0 * (Iv(f(0, x, v)) - 1)]

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

#=
(Expr[:((cord, var"##θ#257", phi, derivative, integral, u, p)->begin
          #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
          #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
          begin
              (var"##θ#2571", var"##θ#2572") = (var"##θ#257"[1:353], var"##θ#257"[354:690])
              (phi1, phi2) = (phi[1], phi[2])
              let (t, x, v) = (cord[[1], :], cord[[2], :], cord[[3], :])
                  begin
                      cord2 = vcat(t, x)
                      cord1 = vcat(t, x, v)
                  end
                  derivative(phi1, u, cord1, [[6.0554544523933395e-6, 0.0, 0.0]], 1, var"##θ#2571") .- (+).((*).(-1, v, derivative(phi1, u, cord1, [[0.0, 6.0554544523933395e-6, 0.0]], 1, var"##θ#2571")), (*).(-1.7588203624634955e11, derivative(phi1, u, cord1, [[0.0, 0.0, 6.0554544523933395e-6]], 1, var"##θ#2571"), u(cord2, var"##θ#2572", phi2)))
              end
          end
      end), :((cord, var"##θ#257", phi, derivative, integral, u, p)->begin
          #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
          #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
          begin
              (var"##θ#2571", var"##θ#2572") = (var"##θ#257"[1:353], var"##θ#257"[354:690])
              (phi1, phi2) = (phi[1], phi[2])
              let (t, x, v) = (cord[[1], :], cord[[2], :], cord[[3], :])
                  begin
                      cord2 = vcat(t, x)
                      cord1 = vcat(t, x, v)
                  end
                  derivative(phi2, u, cord2, [[0.0, 6.0554544523933395e-6]], 1, var"##θ#2572") .- (+).(-1.8095128179727827e-8, (*).(1.8095128179727827e-8, integral(u, cord1, phi, [3], RuntimeGeneratedFunctions.RuntimeGeneratedFunction{(:cord, Symbol("##θ#257"), :phi, :derivative, :integral, :u, :p), NeuralPDE.var"#_RGF_ModTag", NeuralPDE.var"#_RGF_ModTag", (0x8911603e, 0x276c7ab6, 0xa0959381, 0xaec48eb9, 0x95ff9037)}(quote
    #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
    #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
    begin
        (var"##θ#2571", var"##θ#2572") = (var"##θ#257"[1:353], var"##θ#257"[354:690])
        (phi1, phi2) = (phi[1], phi[2])
        let (t, x, v) = (cord[[1], :], cord[[2], :], cord[[3], :])
            begin
                cord1 = vcat(t, x, v)
            end
            u(cord1, var"##θ#2571", phi1)
        end
    end
end), Any[-Inf], Any[Inf], var"##θ#257")))
              end
          end
      end)], Expr[:((cord, var"##θ#257", phi, derivative, integral, u, p)->begin
          #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
          #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
          begin
              (var"##θ#2571", var"##θ#2572") = (var"##θ#257"[1:353], var"##θ#257"[354:690])
              (phi1, phi2) = (phi[1], phi[2])
              let (t, x, v) = (fill(0, size(cord[[1], :])), cord[[1], :], cord[[2], :])
                  begin
                      cord1 = vcat(t, x, v)
                  end
                  u(cord1, var"##θ#2571", phi1) .- (*).(0.28209479177387814, (exp).((*).(-0.24999999999999994, (^).(v, 2))))
              end
          end
      end), :((cord, var"##θ#257", phi, derivative, integral, u, p)->begin
          #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
          #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
          begin
              (var"##θ#2571", var"##θ#2572") = (var"##θ#257"[1:353], var"##θ#257"[354:690])
              (phi1, phi2) = (phi[1], phi[2])
              let (t, x, v) = (fill(0, size(cord[[1], :])), cord[[1], :], cord[[2], :], fill(0, size(cord[[1], :])))
                  begin
                      cord2 = vcat(t, x)
                      cord1 = vcat(t, x, v)
                  end
                  u(cord2, var"##θ#2572", phi2) .- (+).(-1.8095128179727827e-8, (*).(1.8095128179727827e-8, integral(u, cord1, phi, [3], RuntimeGeneratedFunctions.RuntimeGeneratedFunction{(:cord, Symbol("##θ#257"), :phi, :derivative, :integral, :u, :p), NeuralPDE.var"#_RGF_ModTag", NeuralPDE.var"#_RGF_ModTag", (0x8911603e, 0x276c7ab6, 0xa0959381, 0xaec48eb9, 0x95ff9037)}(quote
    #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
    #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:583 =#
    begin
        (var"##θ#2571", var"##θ#2572") = (var"##θ#257"[1:353], var"##θ#257"[354:690])
        (phi1, phi2) = (phi[1], phi[2])
        let (t, x, v) = (cord[[1], :], cord[[2], :], cord[[3], :])
            begin
                cord1 = vcat(t, x, v)
            end
            u(cord1, var"##θ#2571", phi1)
        end
    end
end), Any[-Inf], Any[Inf], var"##θ#257")))
              end
          end
      end)])
=#

cb = function (p, l)
    println("Current loss is: $l")
    return false
end

opt = Optim.BFGS()
res = GalacticOptim.solve(prob, opt, cb = cb, maxiters = 5)
phi = discretization.phi

