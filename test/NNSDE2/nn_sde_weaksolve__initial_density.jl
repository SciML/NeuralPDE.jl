using NeuralPDE, Lux, Distributions, SciMLBase, Statistics, Test
using Random: AbstractRNG
using SymbolicIndexingInterface: getu

# Regression test for SciML/NeuralPDE.jl#1156: SDEPINN must enforce the initial density
# on the whole spatial domain, not only at `u0`.
#
# For f = 0 and g = 1 on [-1, 1] with zero-flux boundaries, every member of
#     p(x, t) = 1/2 + a sin(πx/2) exp(-π²t/8)
# solves the Fokker-Planck equation, integrates to one and has p(0, 0) = 1/2, so a
# condition at u0 = 0 alone cannot tell them apart. Only a = 0 matches Uniform(-1, 1).
struct InitialDensityCounterexample <: Lux.AbstractLuxLayer
    a::Float64
end

Lux.initialparameters(::AbstractRNG, m::InitialDensityCounterexample) = (a = [m.a],)

function (m::InitialDensityCounterexample)(xt, ps, st)
    x, t = xt[1:1, :], xt[2:2, :]
    return 0.5 .+ ps.a[1] .* sin.(π .* x ./ 2) .* exp.(-π^2 .* t ./ 8), st
end

@testset "SDEPINN enforces the initial density on the whole spatial domain" begin
    prob = SDEProblem((u, p, t) -> 0.0, (u, p, t) -> 1.0, 0.0, (0.0, 1.0))
    xs = -1.0:0.1:1.0

    function initial_condition(a)
        alg = SDEPINN(;
            chain = InitialDensityCounterexample(a), x_0 = -1.0, x_end = 1.0,
            distrib = Uniform(-1, 1), dx = step(xs),
        )
        pdesys = NeuralPDE._sdepinn_pdesystem(prob, alg)
        disc = PhysicsInformedNN(alg.chain, GridTraining([alg.dx, 1 / alg.Nt]))
        opt_prob = discretize(pdesys, disc; adtype = alg.adtype)
        # The boundary conditions in T are the zero-flux conditions; the remaining one is
        # the initial condition.
        md = pinn_metadata(opt_prob)
        ic = only(filter(b -> b.kind === :bc && !any(iv -> string(iv) == "T", b.ivs), md.blocks))
        return ic, mean(abs2, getu(opt_prob, ic.residual)(opt_prob))
    end

    ic, loss_exact = initial_condition(0.0)
    @test string.(ic.ivs) == ["X"]
    @test ic.npoints == length(xs)
    @test loss_exact < 1.0e-20

    _, loss_counterexample = initial_condition(0.1)
    @test loss_counterexample > 1.0e-3
end
