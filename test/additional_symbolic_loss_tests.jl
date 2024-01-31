using NeuralPDE: DomainSets
using Random
using Test
using ComponentArrays
using OptimizationOptimisers
using NeuralPDE
using LinearAlgebra
using Lux
import ModelingToolkit: Interval

@parameters x0 x1 x2 x3
@variables ρ01(..) ρ02(..) ρ03(..) ρ12(..) ρ13(..) ρ23(..)

# the 4-torus
domain = [
    x0 ∈ Interval(0.0, 1.0),
    x1 ∈ Interval(0.0, 1.0),
    x2 ∈ Interval(0.0, 1.0),
    x3 ∈ Interval(0.0, 1.0),
]

∂₀ = Differential(x0)
∂₁ = Differential(x1)
∂₂ = Differential(x2)
∂₃ = Differential(x3)

d₂(ρ) = [
    # commented are the signed permutations of the indeces

    #(0,1,2) + (2,0,1) + (1,2,0) - (1,0,2) - (2,1,0) - (0,2,1)
    2 * ∂₀(ρ[4]) - 2 * ∂₁(ρ[2]) + 2 * ∂₂(ρ[1]),
    #(0,1,3) + (3,0,1) + (1,3,0) - (1,0,3) - (0,3,1) - (3,1,0)
    2 * ∂₀(ρ[5]) - 2 * ∂₁(ρ[3]) + 2 * ∂₃(ρ[1]),
    #(0,2,3) + (3,0,2) + (2,3,0) - (2,0,3) - (0,3,2) - (3,2,0)
    2 * ∂₀(ρ[6]) - 2 * ∂₂(ρ[3]) + 2 * ∂₃(ρ[2]),
    #(1,2,3) + (3,1,2) + (2,3,1) - (2,1,3) - (1,3,2) - (3,2,1)
    2 * ∂₁(ρ[6]) - 2 * ∂₂(ρ[5]) + 2 * ∂₃(ρ[4]),
]

u(ρ) = ρ[1] * ρ[6] - ρ[2] * ρ[5] + ρ[3] * ρ[4]

K₁(ρ) = 2(ρ[1] + ρ[6]) / u(ρ)
K₂(ρ) = 2(ρ[2] - ρ[5]) / u(ρ)
K₃(ρ) = 2(ρ[3] + ρ[4]) / u(ρ)

K(ρ) = [
    K₁(ρ),
    K₂(ρ),
    K₃(ρ),
]

# energy
fₑ(ρ) = (K(ρ)[1]^2 + K(ρ)[2]^2 + K(ρ)[3]^2) * u(ρ)

energies = 
    let ρ = [ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)]
        [fₑ(ρ)]
    end

# periodic boundary conditions for the 4-torus
bcs = [
    ρ01(0.0, x1, x2, x3) ~ ρ01(1.0, x1, x2, x3),
    ρ01(x0, 0.0, x2, x3) ~ ρ01(x0, 1.0, x2, x3),
    ρ01(x0, x1, 0.0, x3) ~ ρ01(x0, x1, 1.0, x3),
    ρ01(x0, x1, x2, 0.0) ~ ρ01(x0, x1, x2, 1.0),
    ρ02(0.0, x1, x2, x3) ~ ρ02(1.0, x1, x2, x3),
    ρ02(x0, 0.0, x2, x3) ~ ρ02(x0, 1.0, x2, x3),
    ρ02(x0, x1, 0.0, x3) ~ ρ02(x0, x1, 1.0, x3),
    ρ02(x0, x1, x2, 0.0) ~ ρ02(x0, x1, x2, 1.0),
    ρ03(0.0, x1, x2, x3) ~ ρ03(1.0, x1, x2, x3),
    ρ03(x0, 0.0, x2, x3) ~ ρ03(x0, 1.0, x2, x3),
    ρ03(x0, x1, 0.0, x3) ~ ρ03(x0, x1, 1.0, x3),
    ρ03(x0, x1, x2, 0.0) ~ ρ03(x0, x1, x2, 1.0),
    ρ12(0.0, x1, x2, x3) ~ ρ12(1.0, x1, x2, x3),
    ρ12(x0, 0.0, x2, x3) ~ ρ12(x0, 1.0, x2, x3),
    ρ12(x0, x1, 0.0, x3) ~ ρ12(x0, x1, 1.0, x3),
    ρ12(x0, x1, x2, 0.0) ~ ρ12(x0, x1, x2, 1.0),
    ρ13(0.0, x1, x2, x3) ~ ρ13(1.0, x1, x2, x3),
    ρ13(x0, 0.0, x2, x3) ~ ρ13(x0, 1.0, x2, x3),
    ρ13(x0, x1, 0.0, x3) ~ ρ13(x0, x1, 1.0, x3),
    ρ13(x0, x1, x2, 0.0) ~ ρ13(x0, x1, x2, 1.0),
    ρ23(0.0, x1, x2, x3) ~ ρ23(1.0, x1, x2, x3),
    ρ23(x0, 0.0, x2, x3) ~ ρ23(x0, 1.0, x2, x3),
    ρ23(x0, x1, 0.0, x3) ~ ρ23(x0, x1, 1.0, x3),
    ρ23(x0, x1, x2, 0.0) ~ ρ23(x0, x1, x2, 1.0),
]

# equations for dρ = 0.
eqClosed(ρ) = d₂(ρ)[:] .~ 0

eqs =
    let ρ = [ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)]
        vcat(
            eqClosed(ρ),
        )
    end


input_ = length(domain)
n = 16

ixToSym = Dict(
    1 => :ρ01,
    2 => :ρ02,
    3 => :ρ03,
    4 => :ρ12,
    5 => :ρ13,
    6 => :ρ23
)

chains = NamedTuple((ixToSym[ix], Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1))) for ix in 1:6)
chains0 = collect(chains)

function test_donaldson_energy_loss_no_logs(ϵ, sym_prob, prob) 
    # pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
    # bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions
    # energy_inner_loss_functions = sym_prob.loss_functions.asl_loss_functions

    ps = map(c -> Lux.setup(Random.default_rng(), c)[1], chains) |> ComponentArray .|> Float64
    prob1 = remake(prob; u0 = ComponentVector(depvar = ps))

    callback(ϵ::Float64) = function(p, l)
        # println("loss: ", l)
        # println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
        # println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
        # println("energy losses: ", map(l_ -> l_(p), energy_inner_loss_functions))
        return l < ϵ
    end
    _sol = Optimization.solve(prob1, Adam(0.01); callback=callback(ϵ), maxiters = 1)
    return true
end


@named pdesystem = PDESystem(eqs, bcs, domain, [x0, x1, x2, x3],
    [ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)]
)
discretization = PhysicsInformedNN(chains0, QuasiRandomTraining(1000))
sym_prob = symbolic_discretize(pdesystem, discretization)
prob = discretize(pdesystem, discretization)
@info "testing additional symbolic loss functions: solver runs without additional symbolic losses."
@test test_donaldson_energy_loss_no_logs(0.5, sym_prob, prob)


@named pdesystem1 = PDESystem([], bcs, domain, [x0, x1, x2, x3],
    [ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)]
)
discretization = PhysicsInformedNN(chains0, QuasiRandomTraining(1000); additional_symb_loss = energies)
sym_prob = symbolic_discretize(pdesystem1, discretization)
prob = discretize(pdesystem1, discretization)
@info "testing additional symbolic loss functions: quasi random training: solver runs with only additional symbolic loss function."
@test test_donaldson_energy_loss_no_logs(0.5, sym_prob, prob)

@named pdesystem2 = PDESystem(eqs, bcs, domain, [x0, x1, x2, x3],
    [ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)]
)
discretization = PhysicsInformedNN(chains0, StochasticTraining(1000); additional_symb_loss = energies)
sym_prob = symbolic_discretize(pdesystem2, discretization)
prob = discretize(pdesystem2, discretization)
@info "testing additional symbolic loss functions: stochastic training: solver runs with additional symbolic loss function and PDE system."
@test test_donaldson_energy_loss_no_logs(0.5, sym_prob, prob)

discretization = PhysicsInformedNN(chains0, GridTraining(0.1); additional_symb_loss = energies)
sym_prob = symbolic_discretize(pdesystem2, discretization)
prob = discretize(pdesystem2, discretization)
@info "testing additional symbolic loss functions: grid training: solver runs with additional symbolic loss function and PDE system."
@test test_donaldson_energy_loss_no_logs(0.5, sym_prob, prob)

discretization = PhysicsInformedNN(chains0, QuadratureTraining(); additional_symb_loss = energies)
sym_prob = symbolic_discretize(pdesystem1, discretization)
prob = discretize(pdesystem1, discretization)
@info "testing additional symbolic loss functions: quadrature training: solver runs additional symbolic loss function and PDE system."
@test test_donaldson_energy_loss_no_logs(0.5, sym_prob, prob)
