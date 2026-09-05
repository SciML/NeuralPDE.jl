using ComponentArrays, DomainSets, Lux, ModelingToolkit, NeuralPDE, Random, SciMLBase,
    Test, Zygote

function linear_integral_parameters(chain)
    params = ComponentArray{Float64}(Lux.initialparameters(Xoshiro(123), chain))
    params.layer_1.weight .= 1
    params.layer_1.bias .= 0
    return params
end

@testset "Integral bound variables" begin
    @parameters t tau
    @variables x(..)
    I = Integral(tau in ClosedInterval(0.0, t))
    J = Integral(tau in ClosedInterval(t / 2, t))
    F = Integral(tau in ClosedInterval(0.0, 1.0))
    Dtau = Differential(tau)
    chain = Chain(Dense(1, 1))
    init_params = linear_integral_parameters(chain)
    cord = reshape([0.2, 0.6, 0.9], 1, :)
    cases = [
        ("convolution", I((t - tau) * x(tau)), t^3 / 6, cord .^ 3 ./ 6),
        ("distinct calls", I(x(tau) + x(t)), 3t^2 / 2, 3 .* cord .^ 2 ./ 2),
        ("shifted argument", I((t - tau) * x(t - tau)), t^3 / 3, cord .^ 3 ./ 3),
        ("fixed limits", F(x(tau)), 1 / 2, fill(1 / 2, size(cord))),
        ("variable limits", J(x(tau)), 3t^2 / 8, 3 .* cord .^ 2 ./ 8),
        ("derivative", I((t - tau) * Dtau(x(tau))), t^2 / 2, cord .^ 2 ./ 2),
        ("no dependent variable", I(t - tau), t^2 / 2, zero(cord)),
    ]
    for strategy in (GridTraining(0.2), QuasiRandomTraining(5), QuadratureTraining())
        for (name, integral, expected, weight_derivative) in cases
            @testset "$(typeof(strategy)): $name" begin
                discretization = PhysicsInformedNN(chain, strategy; init_params)
                @named sys = PDESystem(
                    [integral ~ expected], [x(0.0) ~ 0.0],
                    [t ∈ Interval(0.0, 1.0)], [t], [x(t)]
                )
                rep = symbolic_discretize(sys, discretization)
                θ = rep.flat_init_params
                residual = only(rep.loss_functions.datafree_pde_loss_functions)
                @test residual(cord, θ) ≈ zeros(1, 3) atol = 1.0e-8
                @test rep.loss_functions.full_loss_function(θ, nothing) ≈ 0 atol = 1.0e-14
                @test cord == reshape([0.2, 0.6, 0.9], 1, :)
                gradient = only(Zygote.gradient(p -> sum(residual(cord, p)), θ))
                if iszero(weight_derivative)
                    @test gradient === nothing || iszero(gradient)
                else
                    @test gradient.layer_1.weight[1] ≈ sum(weight_derivative)
                end
            end
        end
    end
end

@testset "Multidimensional integration variables" begin
    @parameters t tau sigma
    @variables x(..)
    I = Integral((tau, sigma) in ProductDomain(ClosedInterval(0.0, t), ClosedInterval(0.0, t)))
    chain = Chain(Dense(1, 1))
    init_params = linear_integral_parameters(chain)
    discretization = PhysicsInformedNN(chain, GridTraining(0.2); init_params)
    @named sys = PDESystem(
        [I((t - tau) * (t - sigma) * x(tau)) ~ t^5 / 12], [x(0.0) ~ 0.0],
        [t ∈ Interval(0.0, 1.0)], [t], [x(t)]
    )
    rep = symbolic_discretize(sys, discretization)
    θ = rep.flat_init_params
    cord = reshape([0.2, 0.6, 0.9], 1, :)
    residual = only(rep.loss_functions.datafree_pde_loss_functions)
    @test residual(cord, θ) ≈ zeros(1, 3) atol = 1.0e-8
    gradient = only(Zygote.gradient(p -> sum(residual(cord, p)), θ))
    @test gradient.layer_1.weight[1] ≈ sum(cord .^ 5) / 12
end

@testset "Integral over networks with different inputs" begin
    @parameters t s tau
    @variables x(..) y(..)
    I = Integral(tau in ClosedInterval(0.0, t))
    chains = [Chain(Dense(n, 1)) for n in (1, 2)]
    init_params = linear_integral_parameters.(chains)
    discretization = PhysicsInformedNN(chains, GridTraining(0.2); init_params)
    @named sys = PDESystem(
        [y(0.0, t) + I(x(tau) + y(s, tau)) ~ 0], [x(0.0) ~ 0.0, y(0.0, s) ~ s],
        [t ∈ Interval(0.0, 1.0), s ∈ Interval(0.0, 1.0)], [t, s], [x(t), y(t, s)]
    )
    rep = symbolic_discretize(sys, discretization)
    θ = rep.flat_init_params
    cord = [0.2 0.6 0.9; 0.3 0.5 0.7]
    residual = only(rep.loss_functions.datafree_pde_loss_functions)
    @test residual(cord, θ) ≈ cord[[1], :] .* (cord[[2], :] .+ cord[[1], :] .+ 1)
    grid = 0:0.2:1
    @test rep.loss_functions.full_loss_function(θ, nothing) ≈
        sum(abs2, [t * (s + t + 1) for t in grid, s in grid]) / length(grid)^2
    gradient = only(Zygote.gradient(p -> sum(residual(cord, p)), θ))
    @test gradient.depvar.x.layer_1.weight[1] ≈ sum(cord[1, :] .^ 2) / 2
    @test vec(gradient.depvar.y.layer_1.weight) ≈ [sum(cord[1, :] .* cord[2, :]), sum(cord[1, :] .^ 2) / 2 + sum(cord[1, :])]
end

@testset "Free variables in shifted integrand arguments" begin
    @parameters t tau
    @variables x(..)
    I = Integral(tau in ClosedInterval(0.0, 1.0))
    chain = Chain(Dense(1, 1))
    discretization = PhysicsInformedNN(chain, GridTraining(0.2); init_params = linear_integral_parameters(chain))
    @named sys = PDESystem(
        [I(x(t - tau)) ~ 0], [x(0.0) ~ 0.0],
        [t ∈ Interval(0.0, 1.0)], [t], [x(t)]
    )
    rep = symbolic_discretize(sys, discretization)
    θ = rep.flat_init_params
    cord = reshape([0.2, 0.6, 0.9], 1, :)
    residual = only(rep.loss_functions.datafree_pde_loss_functions)
    @test rep.pde_indvars == [[:t]]
    @test residual(cord, θ) ≈ cord .- 1 / 2
    @test rep.loss_functions.full_loss_function(θ, nothing) ≈ sum(abs2, (0:0.2:1) .- 1 / 2) / 6
    gradient = only(Zygote.gradient(p -> sum(residual(cord, p)), θ))
    @test gradient.layer_1.weight[1] ≈ sum(cord .- 1 / 2)
end
