include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))
using JLArrays, Zygote, LinearAlgebra
using SymbolicIndexingInterface: getp

@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sinpi(x) * sinpi(y)
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]
domains = [x ∈ Interval(0.0, 1.0), y ∈ Interval(0.0, 1.0)]
@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
chain = Chain(Dense(2, 4, σ), Dense(4, 1))
disc = PhysicsInformedNN(
    chain, QuadratureTraining(; quadrature_alg = GaussLegendre(n = 3)); rng = Xoshiro(22)
)
prob = discretize(pde_system, disc)
md = pinn_metadata(prob)
device_parameters = [
    b.xs => jl(getp(prob, b.xs)(prob)) for b in md.blocks if b.xs !== nothing
]
append!(
    device_parameters, [
        b.w => jl(getp(prob, b.w)(prob)) for b in md.blocks if b.w !== nothing
    ]
)

JLArrays.allowscalar(false)

@testset "PDE objective and gradient on JLArray" begin
    device_prob = remake(prob; u0 = jl(prob.u0), p = device_parameters)
    @test device_prob.u0 isa JLArray
    @test all(
        getp(device_prob, b.xs)(device_prob) isa JLArray for b in md.blocks if b.xs !== nothing
    )
    @test all(
        getp(device_prob, b.w)(device_prob) isa JLArray for b in md.blocks if b.w !== nothing
    )
    cpu_loss = prob.f(prob.u0, prob.p)
    device_loss = device_prob.f(device_prob.u0, device_prob.p)
    @test device_loss ≈ cpu_loss
    cpu_gradient = Zygote.gradient(θ -> prob.f(θ, prob.p), prob.u0)[1]
    device_gradient = Zygote.gradient(
        θ -> device_prob.f(θ, device_prob.p), device_prob.u0
    )[1]
    @test device_gradient isa JLArray
    @test Array(device_gradient) ≈ cpu_gradient
end

@testset "resampling preserves JLArray storage" begin
    stochastic = PhysicsInformedNN(chain, StochasticTraining(30; bcs_points = 12); rng = Xoshiro(4))
    stochastic_prob = discretize(pde_system, stochastic)
    stochastic_md = pinn_metadata(stochastic_prob)
    points = [
        b.xs => jl(getp(stochastic_prob, b.xs)(stochastic_prob)) for b in stochastic_md.blocks if b.xs !== nothing
    ]
    device_prob = remake(stochastic_prob; u0 = jl(stochastic_prob.u0), p = points)
    resample!(device_prob)
    @test all(
        getp(device_prob, b.xs)(device_prob) isa JLArray for b in stochastic_md.blocks if b.xs !== nothing
    )
end
