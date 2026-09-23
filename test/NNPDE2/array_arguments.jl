include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))
using ComponentArrays
using ModelingToolkitBase: getdefault
using SymbolicIndexingInterface: getu, getp
using DomainSets: ProductDomain
using SymbolicUtils: unwrap

# The manufactured fields below are the same closed-form functions written into the
# equation, so the finite-difference check uses the stencil of `FiniteDifferenceDerivative`
# (`eps(T)^(1 / (2 + order))`) rather than a tolerance fitted to a solve.

function _residuals(prob)
    return [getu(prob, b.residual)(prob) for b in pinn_metadata(prob).blocks]
end

_f64(θ::AbstractVector{<:Number}) = Float64.(θ)
_f64(θ::AbstractVector) = map(_f64, θ)

function _discretize(sys, chain, θ; dx = 0.25, eval_points = 5)
    disc = PhysicsInformedNN(
        chain, GridTraining(dx); init_params = _f64(θ), rng = Xoshiro(1), eval_points
    )
    return discretize(sys, disc)
end

function _fd_derivative(apply, θ, X, direction, order)
    T = eltype(θ)
    ε = eps(T)^(one(T) / (2 + order))
    e = zeros(T, size(X, 1))
    e[direction] = one(T)
    f(Y) = apply(Y, θ)
    if order == 1
        return (f(X .+ ε .* e) .- f(X .- ε .* e)) ./ (2ε)
    else
        return (f(X .+ ε .* e) .+ f(X .- ε .* e) .- 2 .* f(X)) ./ ε^2
    end
end

# An order-2 central difference cancels down to O(ε^2) with ε = eps^(1/4), so the
# noise floor is eps/ε^2 = √eps per evaluation of an O(1) network. Two compilations
# of that stencil (array arguments vs scalar arguments, or the lowered residual vs
# this direct stencil) differ by a few of those units. 100√eps stays far below an
# O(1) residual, which is what a dropped term would produce.
_close_fd2(a, b) = all(isapprox.(a, b; atol = 100 * sqrt(eps(Float64))))

@testset "array arguments pack to the component form" begin
    @parameters x[1:2]
    @variables u(..)
    xs = collect(x)
    D1 = Differential(xs[1])^2
    D2 = Differential(xs[2])^2
    lap = D1(u(x)) + D2(u(x))
    eq = lap ~ -sinpi(xs[1]) * sinpi(xs[2])
    bcs = [
        u([0.0, xs[2]]) ~ 0.0,
        u([1.0, xs[2]]) ~ 0.0,
        u([xs[1], 2.0]) ~ 0.0,
        u([xs[1], 3.0]) ~ 0.0,
    ]
    domains = [xs[1] ∈ Interval(0.0, 1.0), xs[2] ∈ Interval(2.0, 3.0)]
    @named array_sys = PDESystem(eq, bcs, domains, [x], [u(x)])

    @parameters y1 y2
    @variables v(..)
    Dy1 = Differential(y1)^2
    Dy2 = Differential(y2)^2
    comp_eq = Dy1(v(y1, y2)) + Dy2(v(y1, y2)) ~ -sinpi(y1) * sinpi(y2)
    comp_bcs = [
        v(0.0, y2) ~ 0.0, v(1.0, y2) ~ 0.0, v(y1, 2.0) ~ 0.0, v(y1, 3.0) ~ 0.0,
    ]
    comp_domains = [y1 ∈ Interval(0.0, 1.0), y2 ∈ Interval(2.0, 3.0)]
    @named comp_sys = PDESystem(comp_eq, comp_bcs, comp_domains, [y1, y2], [v(y1, y2)])

    chain = Chain(Dense(2, 8, σ), Dense(8, 1))
    θ = collect(ComponentArray(Lux.initialparameters(Xoshiro(0), chain)))
    aprobe = _discretize(array_sys, chain, θ)
    cprob = _discretize(comp_sys, chain, θ)
    ares = _residuals(aprobe)
    cres = _residuals(cprob)
    @test length(ares) == length(cres) == 5
    @test _close_fd2(ares[1], cres[1])
    for (a, c) in zip(ares[2:end], cres[2:end])
        @test a ≈ c rtol = 1.0e-8
    end
    @test aprobe.f(aprobe.u0, aprobe.p) ≈ cprob.f(cprob.u0, cprob.p) rtol = 1.0e-8

    md = pinn_metadata(aprobe)
    net = only(md.networks)
    @test net.groups !== nothing && length(net.groups) == 1
    @test length(only(net.groups).components) == 2
    @test size(getp(aprobe, md.blocks[1].xs)(aprobe), 1) == 2
    @test md.blocks[1].bounds[1] ≈ [0.0, 2.0]
    @test md.blocks[1].bounds[2] ≈ [1.0, 3.0]

    apply = getdefault(net.NN)
    X = getp(aprobe, md.blocks[1].xs)(aprobe)
    manual = _fd_derivative(apply, aprobe.u0, X, 1, 2) .+
        _fd_derivative(apply, aprobe.u0, X, 2, 2) .+
        sinpi.(X[1:1, :]) .* sinpi.(X[2:2, :])
    @test _close_fd2(ares[1], manual)

    prod_domains = [
        unwrap(x) ∈ ProductDomain([Interval(0.0, 1.0), Interval(2.0, 3.0)]),
    ]
    @named prod_sys = PDESystem(eq, bcs, prod_domains, [x], [u(x)])
    pprob = _discretize(prod_sys, chain, θ)
    @test pinn_metadata(pprob).blocks[1].bounds[1] ≈ [0.0, 2.0]
    @test pinn_metadata(pprob).blocks[1].bounds[2] ≈ [1.0, 3.0]
    @test _residuals(pprob)[1] ≈ ares[1] rtol = 1.0e-8

    # Symbolics fuses `Differential(x)(Differential(x)(u))` into `Differential(x, 2)`,
    # the same stencil as `Differential(x)^2`. This is that operator written with
    # `Differential.(collect(x))`.
    Ds = Differential.(xs)
    nested = Ds[1](Ds[1](u(x))) + Ds[2](Ds[2](u(x))) ~ -sinpi(xs[1]) * sinpi(xs[2])
    @named nested_sys = PDESystem(nested, bcs, domains, [x], [u(x)])
    nested_comp = Differential(y1)(Differential(y1)(v(y1, y2))) +
        Differential(y2)(Differential(y2)(v(y1, y2))) ~ -sinpi(y1) * sinpi(y2)
    @named nested_comp_sys = PDESystem(
        nested_comp, comp_bcs, comp_domains, [y1, y2], [v(y1, y2)]
    )
    @test _close_fd2(
        _residuals(_discretize(nested_sys, chain, θ))[1],
        _residuals(_discretize(nested_comp_sys, chain, θ))[1],
    )

    @test_throws ArgumentError _discretize(array_sys, chain, θ; dx = [0.25])
    vprob = _discretize(array_sys, chain, θ; dx = [0.5, 0.5])
    @test size(getp(vprob, pinn_metadata(vprob).blocks[1].xs)(vprob), 2) == 1
end

@testset "gradient, divergence and curl match the component form" begin
    @parameters x[1:2]
    @variables u(..)
    xs = collect(x)
    grad_eqs = [
        Differential(xs[1])(u(x)) ~ π * cospi(xs[1]) * sinpi(xs[2]),
        Differential(xs[2])(u(x)) ~ π * sinpi(xs[1]) * cospi(xs[2]),
    ]
    bcs = [
        u([0.0, xs[2]]) ~ 0.0,
        u([1.0, xs[2]]) ~ 0.0,
        u([xs[1], 0.0]) ~ 0.0,
        u([xs[1], 1.0]) ~ 0.0,
    ]
    domains = [xs[i] ∈ Interval(0.0, 1.0) for i in 1:2]
    @named gsys = PDESystem(grad_eqs, bcs, domains, [x], [u(x)])
    @parameters y1 y2
    @variables v(..)
    gcomp = [
        Differential(y1)(v(y1, y2)) ~ π * cospi(y1) * sinpi(y2),
        Differential(y2)(v(y1, y2)) ~ π * sinpi(y1) * cospi(y2),
    ]
    gbcs = [v(0.0, y2) ~ 0.0, v(1.0, y2) ~ 0.0, v(y1, 0.0) ~ 0.0, v(y1, 1.0) ~ 0.0]
    @named gcomp_sys = PDESystem(
        gcomp, gbcs, [y1 ∈ Interval(0.0, 1.0), y2 ∈ Interval(0.0, 1.0)], [y1, y2], [v(y1, y2)]
    )
    chain = Chain(Dense(2, 8, σ), Dense(8, 1))
    θ = collect(ComponentArray(Lux.initialparameters(Xoshiro(0), chain)))
    gprob = _discretize(gsys, chain, θ)
    gcprob = _discretize(gcomp_sys, chain, θ)
    gres = _residuals(gprob)
    @test gres ≈ _residuals(gcprob) rtol = 1.0e-8
    apply = getdefault(only(pinn_metadata(gprob).networks).NN)
    X = getp(gprob, pinn_metadata(gprob).blocks[1].xs)(gprob)
    src1 = reshape(π .* cospi.(X[1, :]) .* sinpi.(X[2, :]), 1, :)
    src2 = reshape(π .* sinpi.(X[1, :]) .* cospi.(X[2, :]), 1, :)
    @test gres[1] ≈ _fd_derivative(apply, gprob.u0, X, 1, 1) .- src1
    X2 = getp(gprob, pinn_metadata(gprob).blocks[2].xs)(gprob)
    src2b = reshape(π .* sinpi.(X2[1, :]) .* cospi.(X2[2, :]), 1, :)
    @test gres[2] ≈ _fd_derivative(apply, gprob.u0, X2, 2, 1) .- src2b

    @variables p(..) q(..)
    div_eq = Differential(xs[1])(p(x)) + Differential(xs[2])(q(x)) ~
        2π * cospi(xs[1]) * cospi(xs[2])
    div_bcs = [
        p([0.0, xs[2]]) ~ 0.0, p([1.0, xs[2]]) ~ 0.0,
        q([xs[1], 0.0]) ~ 0.0, q([xs[1], 1.0]) ~ 0.0,
    ]
    @named dsys = PDESystem(div_eq, div_bcs, domains, [x], [p(x), q(x)])
    @variables p2(..) q2(..)
    dcomp = Differential(y1)(p2(y1, y2)) + Differential(y2)(q2(y1, y2)) ~
        2π * cospi(y1) * cospi(y2)
    dbcs = [
        p2(0.0, y2) ~ 0.0, p2(1.0, y2) ~ 0.0, q2(y1, 0.0) ~ 0.0, q2(y1, 1.0) ~ 0.0,
    ]
    @named dcomp_sys = PDESystem(
        dcomp, dbcs, [y1 ∈ Interval(0.0, 1.0), y2 ∈ Interval(0.0, 1.0)],
        [y1, y2], [p2(y1, y2), q2(y1, y2)]
    )
    chains = [chain, chain]
    θs = [
        collect(ComponentArray(Lux.initialparameters(Xoshiro(1), chain))),
        collect(ComponentArray(Lux.initialparameters(Xoshiro(2), chain))),
    ]
    dprob = _discretize(dsys, chains, θs)
    dcprob = _discretize(dcomp_sys, chains, θs)
    dres = _residuals(dprob)
    @test dres[1] ≈ _residuals(dcprob)[1] rtol = 1.0e-8
    md = pinn_metadata(dprob)
    Xd = getp(dprob, md.blocks[1].xs)(dprob)
    apply_p = getdefault(md.networks[1].NN)
    apply_q = getdefault(md.networks[2].NN)
    θp = getu(dprob, md.networks[1].θ)(dprob)
    θq = getu(dprob, md.networks[2].θ)(dprob)
    manual_div = _fd_derivative(apply_p, θp, Xd, 1, 1) .+
        _fd_derivative(apply_q, θq, Xd, 2, 1) .-
        reshape(2π .* cospi.(Xd[1, :]) .* cospi.(Xd[2, :]), 1, :)
    @test dres[1] ≈ manual_div

    @parameters z[1:3]
    @variables a(..) b(..) c(..)
    zs = collect(z)
    # F = (sin(π y), sin(π z), sin(π x)), curl F = -π (cos(π z), cos(π x), cos(π y))
    curl_eqs = [
        Differential(zs[2])(c(z)) - Differential(zs[3])(b(z)) ~ -π * cospi(zs[3]),
        Differential(zs[3])(a(z)) - Differential(zs[1])(c(z)) ~ -π * cospi(zs[1]),
        Differential(zs[1])(b(z)) - Differential(zs[2])(a(z)) ~ -π * cospi(zs[2]),
    ]
    zdomains = [zs[i] ∈ Interval(0.0, 1.0) for i in 1:3]
    # An empty boundary list makes PDEBase call `union()` with no sets. DomainSets
    # adds `union(::Domain...)`, so that call is the empty domain and is not iterable.
    curl_bcs = [a([0.0, zs[2], zs[3]]) ~ 0.0]
    @named csys = PDESystem(curl_eqs, curl_bcs, zdomains, [z], [a(z), b(z), c(z)])
    @parameters r s t
    @variables a2(..) b2(..) c2(..)
    curl_comp = [
        Differential(s)(c2(r, s, t)) - Differential(t)(b2(r, s, t)) ~ -π * cospi(t),
        Differential(t)(a2(r, s, t)) - Differential(r)(c2(r, s, t)) ~ -π * cospi(r),
        Differential(r)(b2(r, s, t)) - Differential(s)(a2(r, s, t)) ~ -π * cospi(s),
    ]
    @named ccomp_sys = PDESystem(
        curl_comp, [a2(0.0, s, t) ~ 0.0],
        [r ∈ Interval(0.0, 1.0), s ∈ Interval(0.0, 1.0), t ∈ Interval(0.0, 1.0)],
        [r, s, t], [a2(r, s, t), b2(r, s, t), c2(r, s, t)]
    )
    chain3 = Chain(Dense(3, 8, σ), Dense(8, 1))
    θ3 = [
        collect(ComponentArray(Lux.initialparameters(Xoshiro(i), chain3))) for i in 1:3
    ]
    chains3 = [chain3, chain3, chain3]
    curl_prob = _discretize(csys, chains3, θ3; dx = 0.5)
    curl_comp_prob = _discretize(ccomp_sys, chains3, θ3; dx = 0.5)
    curl_res = _residuals(curl_prob)
    @test curl_res[1:3] ≈ _residuals(curl_comp_prob)[1:3] rtol = 1.0e-8
    @test curl_res[4] ≈ _residuals(curl_comp_prob)[4] rtol = 1.0e-8
    cmd = pinn_metadata(curl_prob)
    Xc = getp(curl_prob, cmd.blocks[1].xs)(curl_prob)
    applies = [getdefault(net.NN) for net in cmd.networks]
    θc = [getu(curl_prob, net.θ)(curl_prob) for net in cmd.networks]
    # Component order of z is (x, y, z) = (zs[1], zs[2], zs[3]) = directions 1, 2, 3.
    manual_curl = [
        _fd_derivative(applies[3], θc[3], Xc, 2, 1) .-
            _fd_derivative(applies[2], θc[2], Xc, 3, 1) .+ reshape(π .* cospi.(Xc[3, :]), 1, :),
        _fd_derivative(applies[1], θc[1], Xc, 3, 1) .-
            _fd_derivative(applies[3], θc[3], Xc, 1, 1) .+ reshape(π .* cospi.(Xc[1, :]), 1, :),
        _fd_derivative(applies[2], θc[2], Xc, 1, 1) .-
            _fd_derivative(applies[1], θc[1], Xc, 2, 1) .+ reshape(π .* cospi.(Xc[2, :]), 1, :),
    ]
    @test curl_res[1:3] ≈ manual_curl
end

@testset "solution interface for a packed array argument" begin
    @parameters t x[1:2]
    @variables u(..)
    xs = collect(x)
    eq = Differential(t)(u(t, x)) ~ -u(t, x)
    bcs = [u(0.0, x) ~ sinpi(xs[1]) * sinpi(xs[2])]
    domains = [
        t ∈ Interval(0.0, 1.0), xs[1] ∈ Interval(0.0, 1.0), xs[2] ∈ Interval(0.0, 1.0),
    ]
    @named sys = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
    chain = Chain(Dense(3, 8, σ), Dense(8, 1))
    θ = collect(ComponentArray(Lux.initialparameters(Xoshiro(0), chain)))
    prob = _discretize(sys, chain, θ; eval_points = 4)
    sol = solve(prob, Adam(0.01); maxiters = 1)
    @test sol isa PDENoTimeSolution
    expanded = u(t, xs[1], xs[2])
    @test size(sol[u(t, x)]) == (4, 4, 4)
    @test sol[u(t, x)] == sol[expanded]
    @test sol[u(t, x), 2, 3, 4] == sol[u(t, x)][2, 3, 4]
    @test sol(0.3, [0.2, 0.4]; dv = u(t, x)) ≈ sol(0.3, 0.2, 0.4; dv = expanded)
    @test sol(0.3, [0.2, 0.4]; dv = u(t, x)) ≈ sol(0.3, 0.2, 0.4; dv = u(t, x))
    X = [0.2 0.8; 0.4 0.1]
    @test sol(0.3, X; dv = u(t, x)) ≈ [
        sol(0.3, 0.2, 0.4; dv = u(t, x)), sol(0.3, 0.8, 0.1; dv = u(t, x)),
    ]
    tgrid = 0:0.5:1
    @test vec(sol(tgrid, [0.2, 0.4]; dv = u(t, x))) ≈
        [sol(ti, [0.2, 0.4]; dv = u(t, x)) for ti in tgrid]
    @test sol[xs[1]] == collect(range(0.0, 1.0; length = 4))
    net = only(pinn_metadata(prob).networks)
    @test length(net.groups) == 2
    @test length(net.groups[1].components) == 1
    @test length(net.groups[2].components) == 2
end

@testset "manufactured Laplacian and whole-array u(t, z)" begin
    @parameters x[1:2]
    @variables u(..)
    xs = collect(x)
    D1 = Differential(xs[1])^2
    D2 = Differential(xs[2])^2
    eq = D1(u(x)) + D2(u(x)) ~ -sinpi(xs[1]) * sinpi(xs[2])
    bcs = [
        u([0.0, xs[2]]) ~ 0.0,
        u([1.0, xs[2]]) ~ 0.0,
        u([xs[1], 0.0]) ~ 0.0,
        u([xs[1], 1.0]) ~ 0.0,
    ]
    domains = [xs[i] ∈ Interval(0.0, 1.0) for i in 1:2]
    @named sys = PDESystem(eq, bcs, domains, [x], [u(x)])
    # Same manufactured Poisson problem and acceptance bound as test/NNPDE1/poisson_2d.jl.
    analytic(x1, x2) = sinpi(x1) * sinpi(x2) / (2π^2)
    chain = Chain(Dense(2, 12, σ), Dense(12, 12, σ), Dense(12, 1))
    disc = PhysicsInformedNN(chain, GridTraining(0.1); rng = Xoshiro(1), eval_points = 5)
    prob = discretize(sys, disc)
    sol = train(prob)
    xsamp = 0:0.05:1
    err = maximum(
        abs(sol([x1, x2]; dv = u(x)) - analytic(x1, x2)) for x1 in xsamp, x2 in xsamp
    )
    println("Laplacian max abs error ", err)
    @test err < 0.01

    @parameters t z[1:2]
    @variables w(..)
    zs = collect(z)
    # u = exp(-t) sin(π z1) sin(π z2) solves u_t = Δu / (2π^2).
    Dz1 = Differential(zs[1])^2
    Dz2 = Differential(zs[2])^2
    heat = Differential(t)(w(t, z)) ~ (Dz1(w(t, z)) + Dz2(w(t, z))) / (2π^2)
    heat_bcs = [
        w(0.0, z) ~ sinpi(zs[1]) * sinpi(zs[2]),
        w(t, [0.0, zs[2]]) ~ 0.0,
        w(t, [1.0, zs[2]]) ~ 0.0,
        w(t, [zs[1], 0.0]) ~ 0.0,
        w(t, [zs[1], 1.0]) ~ 0.0,
    ]
    heat_domains = [
        t ∈ Interval(0.0, 0.1), zs[1] ∈ Interval(0.0, 1.0), zs[2] ∈ Interval(0.0, 1.0),
    ]
    @named heat_sys = PDESystem(heat, heat_bcs, heat_domains, [t, z], [w(t, z)])
    heat_chain = Chain(Dense(3, 16, σ), Dense(16, 16, σ), Dense(16, 1))
    # Time interval length 0.1: a uniform spacing of 0.1 leaves a single interior
    # time after the initial condition is removed, so the time step is finer.
    heat_disc = PhysicsInformedNN(
        heat_chain, GridTraining([0.025, 0.1, 0.1]); rng = Xoshiro(1), eval_points = 4
    )
    heat_prob = discretize(heat_sys, heat_disc)
    heat_sol = train(heat_prob)
    heat_analytic(tt, z1, z2) = exp(-tt) * sinpi(z1) * sinpi(z2)
    herr = maximum(
        abs(heat_sol(tt, [z1, z2]; dv = w(t, z)) - heat_analytic(tt, z1, z2))
            for tt in (0.0, 0.05, 0.1), z1 in xsamp, z2 in xsamp
    )
    println("heat max abs error ", herr)
    @test herr < 0.05
end
