using ComponentArrays: ComponentArrays, ComponentArray
using Enzyme, ForwardDiff, Lux, LuxCore, NeuralPDE, Random, Test

function dgm_loss(θ, model, x, axes)
    return sum(abs2, LuxCore.stateless_apply(model, x, ComponentArray(θ, axes)))
end

@testset "DGM static reverse with constant input, depth $depth" for depth in (1, 2)
    model = NeuralPDE.DGM(2, 1, 16, depth, tanh, tanh, identity)
    ps, st = Lux.setup(Xoshiro(42), model)
    params = ComponentArray{Float64}(ps)
    axes = ComponentArrays.getaxes(params)
    θ = collect(params)
    x = [0.1 0.2 0.4; 0.3 0.4 0.5]
    @test LuxCore.stateless_apply(model, x, params) == first(Lux.apply(model, x, params, st))
    expected = ForwardDiff.gradient(p -> dgm_loss(p, model, x, axes), θ)
    gradient = zero(θ)
    Enzyme.autodiff(
        Enzyme.Reverse, dgm_loss, Enzyme.Active, Enzyme.Duplicated(θ, gradient),
        Enzyme.Const(model), Enzyme.Const(x), Enzyme.Const(axes)
    )
    @test gradient ≈ expected
end
