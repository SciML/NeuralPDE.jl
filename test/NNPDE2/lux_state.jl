include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))
using ADTypes: AutoEnzyme, AutoZygote
using Enzyme: Enzyme
using Boltz.Layers: PeriodicEmbedding
using ComponentArrays: ComponentArray, getaxes

function periodic_problem()
    @parameters x
    @variables u(..)
    Dx = Differential(x)
    @named pde_system = PDESystem(
        [Dx(u(x)) ~ cos(x)], [u(0.0) ~ 0.0], [x ∈ Interval(0.0, 2π)], [x], [u(x)]
    )
    return pde_system, u(x)
end

periodic_chain() = Chain(PeriodicEmbedding([1], [2π]), Dense(2, 8, tanh), Dense(8, 1))

@testset "stateful Lux layers (issue #1222)" begin
    pde_system, uref = periodic_problem()
    chain = periodic_chain()
    disc = PhysicsInformedNN(chain, QuasiRandomTraining(100); rng = Xoshiro(0))
    prob = discretize(pde_system, disc)
    @test prob.f.adtype isa AutoZygote
    @test length(prob.u0) == Lux.parameterlength(chain)
    sol = train(prob; adam_iters = 500, bfgs_iters = 500)
    xs = range(0.0, 2π; length = 50)
    @test maximum(abs, [sol(xi; dv = uref) - sin(xi) for xi in xs]) < 1.0e-3

    eprob = discretize(
        pde_system, disc; adtype = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    )
    esol = solve(eprob, Adam(0.01); maxiters = 5)
    @test all(isfinite, esol.original_sol.u)
    @test esol.original_sol.u != eprob.u0
end

@testset "user-supplied init_states" begin
    pde_system, uref = periodic_problem()
    chain = periodic_chain()
    ps0 = Lux.initialparameters(Xoshiro(1), chain)
    st = Lux.initialstates(Xoshiro(1), chain)
    st_user = merge(st, (; layer_1 = (; st.layer_1..., k = [2 / π])))
    disc = PhysicsInformedNN(
        chain, GridTraining(0.5); init_params = ps0, init_states = st_user, rng = Xoshiro(0)
    )
    sol = solve(discretize(pde_system, disc), Adam(0.01); maxiters = 1)
    θ = sol.original_sol.u
    ps = ComponentArray(θ, getaxes(ComponentArray(ps0)))
    xs = [0.3, 1.1, 2.5]
    eval_with(s) = vec(first(chain(reshape(eltype(θ).(xs), 1, :), ps, s)))
    @test [sol(xi; dv = uref) for xi in xs] ≈ eval_with(st_user)
    @test !isapprox(eval_with(st_user), eval_with(st))

    @test_throws ArgumentError symbolic_discretize(
        pde_system, PhysicsInformedNN(chain, GridTraining(0.5); init_states = [st, st])
    )
end
