using NeuralPDE, Lux, ComponentArrays, Random, Zygote, Test

@testset "NNODE finite time derivative parameter gradients" begin
    for T in (Float32, Float64), S in (Float32, Float64), scalar in (true, false)
        chain = Chain(Dense(1, 1))
        _, st = Lux.setup(Random.default_rng(), chain)
        theta = ComponentArray(;
            depvar = (; layer_1 = (; weight = zeros(T, 1, 1), bias = ones(T, 1)))
        )
        phi = NeuralPDE.ODEPhi(chain, zero(S), scalar ? zero(S) : zeros(S, 1), st)
        for time in (S(0.4), S[0.4])
            loss = p -> sum(abs2, NeuralPDE.ode_dfdx(phi, time, p, false))
            gradient = only(Zygote.gradient(loss, theta))
            @test gradient.depvar.layer_1.bias ≈ T[2] rtol = 1.0e-3
        end
        rhs = NeuralPDE.BatchedRHS((u, p, t) -> zero(u))
        loss = p -> NeuralPDE.inner_loss(phi, rhs, false, S[0.4], p, nothing, false)
        gradient = only(Zygote.gradient(loss, theta))
        @test gradient.depvar.layer_1.bias ≈ T[2] rtol = 1.0e-3
    end
end
