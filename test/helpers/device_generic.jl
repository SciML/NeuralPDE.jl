using Zygote, ADTypes
using SymbolicIndexingInterface: getp
using Symbolics: build_function
using SymbolicUtils: unwrap, BasicSymbolic, isconst, unwrap_const, iscall, arguments, operation

function has_host_array(ex)
    ex = unwrap(ex)
    if ex isa BasicSymbolic
        isconst(ex) && return has_host_array(unwrap_const(ex))
        return iscall(ex) && any(has_host_array, arguments(ex))
    end
    return ex isa Array || (ex isa Tuple && any(has_host_array, ex))
end

function test_device_pde(dev, DeviceArray)
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
    prob = discretize(pde_system, disc; adtype = AutoZygote())
    md = pinn_metadata(prob)
    device_parameters = [
        b.xs => dev(getp(prob, b.xs)(prob)) for b in md.blocks if b.xs !== nothing
    ]
    append!(
        device_parameters, [
            b.w => dev(getp(prob, b.w)(prob)) for b in md.blocks if b.w !== nothing
        ]
    )


    @testset "PDE objective, gradient and solve on $DeviceArray" begin
        @test !any(b -> has_host_array(b.residual), md.blocks)
        device_prob = remake(prob; u0 = dev(prob.u0), p = device_parameters)
        @test device_prob.u0 isa DeviceArray
        @test eltype(device_prob.u0) === eltype(prob.u0)
        @test all(
            getp(device_prob, b.xs)(device_prob) isa DeviceArray for b in md.blocks if b.xs !== nothing
        )
        @test all(
            getp(device_prob, b.w)(device_prob) isa DeviceArray for b in md.blocks if b.w !== nothing
        )
        cpu_loss = prob.f(prob.u0, prob.p)
        device_loss = device_prob.f(device_prob.u0, device_prob.p)
        @test device_loss ≈ cpu_loss
        cpu_gradient = Zygote.gradient(θ -> prob.f(θ, prob.p), prob.u0)[1]
        device_gradient = Zygote.gradient(
            θ -> device_prob.f(θ, device_prob.p), device_prob.u0
        )[1]
        @test device_gradient isa DeviceArray
        @test Array(device_gradient) ≈ cpu_gradient
        cpu_sol = solve(prob, Adam(0.01); maxiters = 5)
        device_sol = solve(device_prob, Adam(0.01); maxiters = 5)
        @test device_sol.original_sol.u isa DeviceArray
        @test device_sol.original_sol.objective < device_loss
        @test device_sol.original_sol.objective ≈ cpu_sol.original_sol.objective
        @test device_sol[u(x, y)] isa Array
        @test device_sol[u(x, y)] ≈ cpu_sol[u(x, y)]
    end

    @testset "resampling on $DeviceArray" begin
        stochastic = PhysicsInformedNN(chain, StochasticTraining(30; bcs_points = 12); rng = Xoshiro(4))
        stochastic_prob = discretize(pde_system, stochastic; adtype = AutoZygote())
        stochastic_md = pinn_metadata(stochastic_prob)
        points = [
            b.xs => dev(getp(stochastic_prob, b.xs)(stochastic_prob)) for b in stochastic_md.blocks if b.xs !== nothing
        ]
        device_prob = remake(stochastic_prob; u0 = dev(stochastic_prob.u0), p = points)
        before = [Array(getp(device_prob, b.xs)(device_prob)) for b in stochastic_md.blocks if b.xs !== nothing]
        resample!(device_prob; rng = Xoshiro(11))
        resample!(stochastic_prob; rng = Xoshiro(11))
        after = [getp(device_prob, b.xs)(device_prob) for b in stochastic_md.blocks if b.xs !== nothing]
        @test all(Array(a) != b for (a, b) in zip(after, before))
        @test device_prob.f(device_prob.u0, device_prob.p) ≈ stochastic_prob.f(stochastic_prob.u0, stochastic_prob.p)
        cpu_gradient = Zygote.gradient(θ -> stochastic_prob.f(θ, stochastic_prob.p), stochastic_prob.u0)[1]
        device_gradient = Zygote.gradient(θ -> device_prob.f(θ, device_prob.p), device_prob.u0)[1]
        @test device_gradient isa DeviceArray
        @test Array(device_gradient) ≈ cpu_gradient
        @test all(
            getp(device_prob, b.xs)(device_prob) isa DeviceArray for b in stochastic_md.blocks if b.xs !== nothing
        )
    end

    @testset "network arguments on $DeviceArray" begin
        extra_bcs = [
            u(0.0, 0.0) ~ 0.0,
            u(y, x) ~ 0.0,
            u(x, x) ~ 0.0,
            Differential(x)(u(1.0, y)) ~ 0.0,
        ]
        @named arguments_system = PDESystem(eq, extra_bcs, domains, [x, y], [u(x, y)])
        argument_prob = discretize(arguments_system, disc; adtype = AutoZygote())
        argument_md = pinn_metadata(argument_prob)
        @test !any(b -> has_host_array(b.residual), argument_md.blocks)
        @test all(size(b.residual) == (1, b.npoints) for b in argument_md.blocks)
        data = Pair[]
        for b in argument_md.blocks, key in (b.xs, b.w)
            key === nothing || push!(data, key => dev(getp(argument_prob, key)(argument_prob)))
        end
        device_prob = remake(argument_prob; u0 = dev(argument_prob.u0), p = data)
        @test device_prob.f(device_prob.u0, device_prob.p) ≈ argument_prob.f(argument_prob.u0, argument_prob.p)
        cpu_gradient = Zygote.gradient(θ -> argument_prob.f(θ, argument_prob.p), argument_prob.u0)[1]
        device_gradient = Zygote.gradient(θ -> device_prob.f(θ, device_prob.p), device_prob.u0)[1]
        @test device_gradient isa DeviceArray
        @test Array(device_gradient) ≈ cpu_gradient
    end
    @testset "general argument lowering on $DeviceArray" begin
        @parameters a
        b = md.blocks[1]
        net = only(md.networks)
        indices = Dict(unwrap(x) => 1, unwrap(y) => 2)
        ctx = NeuralPDE.LoweringContext(
            unwrap(b.xs), indices, indices, Dict(operation(unwrap(u(x, y))) => net),
            Dict(unwrap(a) => 0.25), disc.derivative, b.npoints, eltype(prob.u0),
            nothing, Any[], :device_arguments
        )
        expr = NeuralPDE.lower(u(x / 2, a), ctx, zeros(eltype(prob.u0), 2))
        @test !has_host_array(expr)
        X = getp(prob, b.xs)(prob)
        apply = getp(prob, net.NN)(prob)
        expected = apply(
            vcat(X[1:1, :] ./ 2, fill(0.25, 1, b.npoints)), prob.u0
        )
        evaluate = build_function(
            unwrap(expr), b.xs, net.θ, net.NN; expression = Val{false}
        )[1]
        cpu_values = evaluate(X, prob.u0, apply)
        @test cpu_values ≈ expected
        device_prob = remake(prob; u0 = dev(prob.u0), p = device_parameters)
        device_X = getp(device_prob, b.xs)(device_prob)
        @test Array(evaluate(device_X, device_prob.u0, apply)) ≈ cpu_values
        loss(θ, X) = NeuralPDE._mean_square(evaluate(X, θ, apply), b.npoints)
        cpu_gradient = Zygote.gradient(θ -> loss(θ, X), prob.u0)[1]
        device_gradient = Zygote.gradient(θ -> loss(θ, device_X), device_prob.u0)[1]
        @test device_gradient isa DeviceArray
        @test Array(device_gradient) ≈ cpu_gradient
    end
    return nothing
end
