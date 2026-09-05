using NeuralPDE, Lux, LuxCore, ComponentArrays, Random, Zygote, Test

@testset "NNSDE finite time derivative preserves parameter gradients" begin
    for T in (Float32, Float64), S in (Float32, Float64)
        chain = Chain(Dense(2, 1))
        st = LuxCore.initialstates(Random.default_rng(), chain)
        theta = ComponentArray(; depvar = (; layer_1 = (; weight = zeros(T, 1, 2), bias = ones(T, 1))))
        phi = NeuralPDE.SDEPhi(chain, zero(S), zero(S), st)
        inputs = reshape(S[0.4, 0], 2, 1)
        loss(p) = sum(abs2, NeuralPDE.∂u_∂t(inputs, (phi, p, false)))
        gradient = only(Zygote.gradient(loss, theta))
        @test gradient.depvar.layer_1.bias ≈ T[2] rtol = 1.0e-3
    end
end
