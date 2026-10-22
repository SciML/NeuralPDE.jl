include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))
using ComponentArrays, ForwardDiff
import LuxCore
using ModelingToolkitBase: getdefault
using SymbolicIndexingInterface: getp
using Boltz.Layers: PeriodicEmbedding

struct CounterStateLayer <: LuxCore.AbstractLuxLayer end
LuxCore.initialparameters(::AbstractRNG, ::CounterStateLayer) = (; weight = fill(0.8, 1, 1))
LuxCore.initialstates(::AbstractRNG, ::CounterStateLayer) = (; counter = Ref(0.0))
function (::CounterStateLayer)(x, ps, st)
    st.counter[] += 1
    return ps.weight * x .+ st.counter[], st
end

struct AliasStateLayer <: LuxCore.AbstractLuxLayer end
LuxCore.initialparameters(::AbstractRNG, ::AliasStateLayer) = NamedTuple()
function LuxCore.initialstates(::AbstractRNG, ::AliasStateLayer)
    a = [0.0]
    return (; counter = a, mirror = a)
end
function (::AliasStateLayer)(x, ps, st)
    st.counter[1] += 1
    return x .+ st.mirror, st
end

struct EmptyCacheLayer{T} <: LuxCore.AbstractLuxLayer
    cache::T
end
LuxCore.initialparameters(::AbstractRNG, ::EmptyCacheLayer) = (; weight = fill(0.8, 1, 1))
LuxCore.initialstates(::AbstractRNG, l::EmptyCacheLayer) = (; cache = l.cache)
function (::EmptyCacheLayer)(x, ps, st)
    @assert st.cache === nothing || isempty(st.cache)
    return ps.weight * x, st
end

function fixed_state_default_gradient(prob)
    gradient = Ref{Vector{eltype(prob.u0)}}()
    solve(
        prob, Adam(0.001); maxiters = 1, callback = function (state, loss)
            gradient[] = copy(state.grad)
            return false
        end
    )
    return gradient[]
end

@testset "fixed Lux state contracts" begin
    @parameters s
    @variables v(..)
    @named field_system = PDESystem(
        [v(s) ~ s], [v(0.0) ~ 0.0], [s ∈ Interval(0.0, 1.0)], [s], [v(s)]
    )

    @testset "stateless RandomDevice" begin
        chain = Chain(Dense(1, 4, tanh), Dense(4, 1))
        prob = discretize(
            field_system, PhysicsInformedNN(chain, GridTraining(0.25); rng = RandomDevice())
        )
        @test isfinite(prob.f(prob.u0, prob.p))
        phi = NeuralPDE.trial_function(only(pinn_metadata(prob).networks), prob.u0)
        ps, st = LuxCore.setup(Xoshiro(1), chain)
        ps = ComponentArray(prob.u0, getaxes(ComponentArray(ps)))
        X = [0.25 0.75]
        @test phi(X) ≈ first(chain(X, ps, st))
        @test !(NeuralPDE.with_fixed_state(chain, Xoshiro(1), Float64) isa NeuralPDE.FixedStateLayer)
    end

    @testset "inference field batching: $layer" for layer in (Dropout(0.5), BatchNorm(8))
        chain = Chain(Dense(1, 8, tanh), layer, Dense(8, 1))
        prob = discretize(
            field_system, PhysicsInformedNN(chain, GridTraining(0.25); rng = Xoshiro(77))
        )
        phi = NeuralPDE.trial_function(only(pinn_metadata(prob).networks), prob.u0)
        ps, st = LuxCore.setup(Xoshiro(77), chain)
        ps = ComponentArray{Float64}(ps)
        X = [0.1 0.7 0.4 0.7]
        expected = first(chain(X, ps, LuxCore.testmode(st)))
        @test phi(X) ≈ expected
        @test phi(fill(0.7, 1, 8)) ≈ fill(only(phi(fill(0.7, 1, 1))), 1, 8)
        @test phi(X) ≈ hcat((phi(X[:, i:i]) for i in axes(X, 2))...)
        loss = prob.f(prob.u0, prob.p)
        @test isfinite(loss)
        @test prob.f(prob.u0, prob.p) == loss
        @test fixed_state_default_gradient(prob) ≈
            ForwardDiff.gradient(w -> prob.f(w, prob.p), prob.u0) rtol = 1.0e-6
        @test phi(X) ≈ expected
    end

    @testset "isolated Ref state" begin
        prob = discretize(
            field_system,
            PhysicsInformedNN(Chain(CounterStateLayer()), GridTraining(0.25); rng = Xoshiro(77))
        )
        md = pinn_metadata(prob)
        net = only(md.networks)
        model = getdefault(net.NN).lux_model
        phi = NeuralPDE.trial_function(net, prob.u0)
        X = [0.1 0.7 0.4]
        expected = 0.8 .* X .+ 1
        @test phi(X) ≈ expected
        @test phi(X) ≈ expected
        xs = vec(getp(prob, md.blocks[1].xs)(prob))
        residual = (0.8 - 1) .* xs .+ 1
        loss = sum(abs2, residual) / length(xs) + 1
        gradient = [2sum(residual .* xs) / length(xs)]
        @test prob.f(prob.u0, prob.p) ≈ loss
        @test prob.f(prob.u0, prob.p) ≈ loss
        @test ForwardDiff.gradient(w -> prob.f(w, prob.p), prob.u0) ≈ gradient
        @test fixed_state_default_gradient(prob) ≈ gradient
        @test model.st.layer_1.counter[] == 0
        @test phi(X) ≈ expected
        @test prob.f(prob.u0, prob.p) ≈ loss
    end

    @testset "required empty cache: $cache" for cache in (nothing, Float64[])
        chain = Chain(EmptyCacheLayer(cache))
        @test NeuralPDE.with_fixed_state(chain, Xoshiro(77), Float64) isa NeuralPDE.FixedStateLayer
        prob = discretize(
            field_system, PhysicsInformedNN(chain, GridTraining(0.25); rng = Xoshiro(77))
        )
        md = pinn_metadata(prob)
        phi = NeuralPDE.trial_function(only(md.networks), prob.u0)
        X = [0.25 0.75]
        @test phi(X) ≈ 0.8 .* X
        xs = vec(getp(prob, md.blocks[1].xs)(prob))
        @test prob.f(prob.u0, prob.p) ≈ (0.8 - 1)^2 * sum(abs2, xs) / length(xs)
        gradient = [2(0.8 - 1) * sum(abs2, xs) / length(xs)]
        @test fixed_state_default_gradient(prob) ≈ gradient
        @test ForwardDiff.gradient(w -> prob.f(w, prob.p), prob.u0) ≈ gradient
    end

    @testset "aliased state buffers" begin
        chain = Chain(AliasStateLayer())
        prob = discretize(
            field_system, PhysicsInformedNN(chain, GridTraining(0.25); rng = Xoshiro(77))
        )
        md = pinn_metadata(prob)
        net = only(md.networks)
        model = getdefault(net.NN).lux_model
        stored = model.st.layer_1
        @test stored.counter === stored.mirror
        @test stored.counter == [0.0]
        wrapper = getdefault(net.NN)
        X = fill(0.25, 1, 1)
        @test wrapper(X, Float64[]) ≈ fill(1.25, 1, 1)
        @test wrapper(X, Float64[]) ≈ fill(1.25, 1, 1)
        @test stored.counter === stored.mirror
        @test stored.counter == [0.0]
        @test model.st.layer_1.counter == [0.0]
    end

    @testset "periodic analytic field and gradients" begin
        Ds = Differential(s)
        @named periodic_system = PDESystem(
            [Ds(v(s)) ~ cos(s)], [v(0.0) ~ 0.0], [s ∈ Interval(0.0, 2π)], [s], [v(s)]
        )
        chain = Chain(PeriodicEmbedding([1], [2π]), Dense(2, 1))
        ps, _ = LuxCore.setup(Xoshiro(77), chain)
        ps = ComponentArray{Float64}(ps)
        a, b, c = 0.8, 0.3, -0.2
        ps.layer_2.weight .= [a b]
        ps.layer_2.bias .= c
        prob = discretize(
            periodic_system,
            PhysicsInformedNN(
                chain, GridTraining(0.25); init_params = ps,
                rng = Xoshiro(77), derivative = EnzymeForwardDerivative()
            )
        )
        md = pinn_metadata(prob)
        phi = NeuralPDE.trial_function(only(md.networks), prob.u0)
        X = reshape(collect(range(0.0, 2π; length = 101)), 1, :)
        @test phi(X) ≈ a .* sin.(X) .+ b .* cos.(X) .+ c
        xs = vec(getp(prob, md.blocks[1].xs)(prob))
        residual = (a - 1) .* cos.(xs) .- b .* sin.(xs)
        loss = sum(abs2, residual) / length(xs) + (b + c)^2
        gradient = [
            2sum(residual .* cos.(xs)) / length(xs),
            -2sum(residual .* sin.(xs)) / length(xs) + 2(b + c), 2(b + c),
        ]
        @test prob.f(prob.u0, prob.p) ≈ loss
        @test ForwardDiff.gradient(w -> prob.f(w, prob.p), prob.u0) ≈ gradient
        @test fixed_state_default_gradient(prob) ≈ gradient
    end
end
