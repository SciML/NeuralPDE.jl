using ModelingToolkit, NeuralPDE, SciMLBase
using Test

@testset "BPINN PDE data quadrature: derivative order" begin
    using Lux, Distributions, AdvancedHMC, LogDensityProblems, MCMCChains,
        ComponentArrays, ForwardDiff, Random, SymbolicUtils
    import DomainSets: Interval

    ext = Base.get_extension(NeuralPDE, :NeuralPDEBPINNExt)

    @parameters x t
    @variables u(..)
    Dx = Differential(x)
    Dt = Differential(t)

    # Symbolics 7 fuses Differential(x)^n into one Differential(x, n) node that
    # carries an `order`; mixed partials stay nested. Both must be accumulated.
    _, orders_fused = ext._diff_spec((Dx^4)(u(x, t)))
    @test orders_fused[Symbolics.unwrap(x)] == 4
    _, orders_mixed = ext._diff_spec(Dx(Dt(u(x, t))))
    @test orders_mixed == Dict{Any, Int}(Symbolics.unwrap(x) => 1, Symbolics.unwrap(t) => 1)
    _, orders_nested = ext._diff_spec(Dx(Dt((Dx^2)(u(x, t)))))
    @test orders_nested == Dict{Any, Int}(Symbolics.unwrap(x) => 3, Symbolics.unwrap(t) => 1)

    # The data-quadrature likelihood at the dataset points, compared against
    # hand-written ForwardDiff references of the same masked residuals.
    # rtol = 1e-4 allows O(ε²) truncation and cancellation for ε = 1e-3.
    function quadrature_likelihood(
            sys, disc, Dict_differentials, dataset, chain, θ, phynewstd
        )
        prob = SciMLBase.discretize(sys, disc.pinn)
        md = NeuralPDE.pinn_metadata(prob)
        nets = NeuralPDE.unique_networks(md.networks)
        network_fns = [
            (X, θi) -> ModelingToolkit.getdefault(net.NN)(X, θi)
                for net in nets
        ]
        net_lengths = [length(ModelingToolkit.getdefault(net.θ)) for net in nets]
        L2 = ext.build_data_quadrature(
            sys, Dict_differentials, dataset, network_fns, net_lengths, md
        )
        return L2(θ, [phynewstd])
    end

    function nn_of(chain, θnn)
        ps0, st = Lux.setup(Random.default_rng(), chain)
        ax = getaxes(ComponentArray(ps0))
        return X -> only(first(chain(X, ComponentArray(θnn, ax), st)))
    end

    @testset "forcing = $forcing" for forcing in (zero, cos)
        @parameters t p
        @variables u(..)
        Dtt = Differential(t)^2
        eq = Dtt(u(t)) ~ -p * u(t) + forcing(t)
        bcs = [u(0.0) ~ 0.0, Differential(t)(u(0.0)) ~ 1.0]
        @named sys = PDESystem(
            eq, bcs, [t ∈ Interval(0.0, 1.0)], [t], [u(t)], [p];
            initial_conditions = Dict(p => 1.0)
        )
        tdata = [0.1, 0.4, 0.7, 0.95]
        ydata = sin.(tdata)
        dataset = [hcat(ydata, tdata)]
        D = Dict(Dtt(u(t)) => Symbolics.variable(:diff_1))
        chain = Lux.Chain(Lux.Dense(1, 4, Lux.tanh), Lux.Dense(4, 1))
        disc = BayesianPINN(
            [chain], GridTraining([0.25]); param_estim = true,
            dataset = [dataset, nothing], rng = Random.Xoshiro(3)
        )
        θ = collect(SciMLBase.discretize(sys, disc.pinn).u0)
        phynewstd = 0.35
        code_dq = quadrature_likelihood(sys, disc, D, dataset, chain, θ, phynewstd)
        θnn = θ[1:(end - 1)]
        pp = θ[end]
        nn = nn_of(chain, θnn)
        hand = sum(
            logpdf(
                Normal(0, phynewstd),
                ForwardDiff.derivative(
                    s -> ForwardDiff.derivative(z -> nn([z]), s), xi
                ) + pp * yi - forcing(xi)
            )
                for (xi, yi) in zip(tdata, ydata)
        )
        @test code_dq ≈ hand rtol = 1.0e-4
    end

    let
        @parameters x t p
        @variables u(..)
        Dx = Differential(x)
        Dt = Differential(t)
        eq = Dx(Dt(u(x, t))) ~ -p * u(x, t)
        bcs = [u(x, 0.0) ~ 0.0]
        @named sys = PDESystem(
            eq, bcs, [x ∈ Interval(0.0, 1.0), t ∈ Interval(0.0, 1.0)],
            [x, t], [u(x, t)], [p]; initial_conditions = Dict(p => 1.0)
        )
        pts = [(0.2, 0.1), (0.5, 0.4), (0.8, 0.9)]
        ydata = [sin(xi + ti) for (xi, ti) in pts]
        dataset = [hcat(ydata, [xi for (xi, ti) in pts], [ti for (xi, ti) in pts])]
        D = Dict(Dx(Dt(u(x, t))) => Symbolics.variable(:diff_1))
        chain = Lux.Chain(Lux.Dense(2, 4, Lux.tanh), Lux.Dense(4, 1))
        disc = BayesianPINN(
            [chain], GridTraining([0.25, 0.25]); param_estim = true,
            dataset = [dataset, nothing], rng = Random.Xoshiro(4)
        )
        θ = collect(SciMLBase.discretize(sys, disc.pinn).u0)
        phynewstd = 0.3
        code_dq = quadrature_likelihood(sys, disc, D, dataset, chain, θ, phynewstd)
        θnn = θ[1:(end - 1)]
        pp = θ[end]
        nn = nn_of(chain, θnn)
        hand = sum(
            logpdf(
                Normal(0, phynewstd),
                ForwardDiff.derivative(
                    xv -> ForwardDiff.derivative(tv -> nn([xv, tv]), ti), xi
                ) + pp * yi
            )
                for ((xi, ti), yi) in zip(pts, ydata)
        )
        @test code_dq ≈ hand rtol = 1.0e-4
    end

    # Fused mixed partial: Dx^2(Dt(u)) = -p u (order 2 in x, 1 in t).
    let
        @parameters x t p
        @variables u(..)
        Dx = Differential(x)
        Dt = Differential(t)
        eq = (Dx^2)(Dt(u(x, t))) ~ -p * u(x, t)
        bcs = [u(x, 0.0) ~ 0.0]
        @named sys = PDESystem(
            eq, bcs, [x ∈ Interval(0.0, 1.0), t ∈ Interval(0.0, 1.0)],
            [x, t], [u(x, t)], [p]; initial_conditions = Dict(p => 1.0)
        )
        pts = [(0.2, 0.1), (0.5, 0.4), (0.8, 0.9)]
        ydata = [cos(xi * ti) for (xi, ti) in pts]
        dataset = [hcat(ydata, [xi for (xi, ti) in pts], [ti for (xi, ti) in pts])]
        D = Dict((Dx^2)(Dt(u(x, t))) => Symbolics.variable(:diff_1))
        chain = Lux.Chain(Lux.Dense(2, 4, Lux.tanh), Lux.Dense(4, 1))
        disc = BayesianPINN(
            [chain], GridTraining([0.25, 0.25]); param_estim = true,
            dataset = [dataset, nothing], rng = Random.Xoshiro(5)
        )
        θ = collect(SciMLBase.discretize(sys, disc.pinn).u0)
        phynewstd = 0.3
        code_dq = quadrature_likelihood(sys, disc, D, dataset, chain, θ, phynewstd)
        θnn = θ[1:(end - 1)]
        pp = θ[end]
        nn = nn_of(chain, θnn)
        hand = sum(
            logpdf(
                Normal(0, phynewstd),
                ForwardDiff.derivative(
                    x1 ->
                    ForwardDiff.derivative(
                        x2 -> ForwardDiff.derivative(tv -> nn([x2, tv]), ti),
                        x1
                    ),
                    xi
                ) + pp * yi
            )
                for ((xi, ti), yi) in zip(pts, ydata)
        )
        @test code_dq ≈ hand rtol = 1.0e-4
    end
end
