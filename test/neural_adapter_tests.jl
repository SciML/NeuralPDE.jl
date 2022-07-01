using Flux, OptimizationFlux
using Test, NeuralPDE
using Optimization, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum
using Statistics

using Random
Random.seed!(100)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

## Example, 2D Poisson equation with Neural adapter
println("Example, 2D Poisson equation with Neural adapter")
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# 2D PDE
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

# Initial and boundary conditions
bcs = [u(0, y) ~ 0.0, u(1, y) ~ -sin(pi * 1) * sin(pi * y),
    u(x, 0) ~ 0.0, u(x, 1) ~ -sin(pi * x) * sin(pi * 1)]
# Space and time domains
domains = [x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)]
quadrature_strategy = NeuralPDE.QuadratureTraining(reltol = 1e-2, abstol = 1e-2,
                                                   maxiters = 50, batch = 100)
inner = 8
af = Flux.tanh
chain1 = Chain(Dense(2, inner, af),
               Dense(inner, inner, af),
               Dense(inner, 1)) |> f64
initθ = Flux.destructure(chain1)[1]
discretization = NeuralPDE.PhysicsInformedNN(chain1,
                                             quadrature_strategy;
                                             init_params = initθ)

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])
prob = NeuralPDE.discretize(pde_system, discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
res = Optimization.solve(prob, BFGS(); maxiters = 2000)
phi = discretization.phi

inner_ = 10
af = Flux.tanh
chain2 = Lux.Chain(Lux.Dense(2, inner_, af),
                   Lux.Dense(inner_, inner_, af),
                   Lux.Dense(inner_, inner_, af),
                   Lux.Dense(inner_, 1))
initθ2 = Float64.(ComponentArray(Lux.setup(Random.default_rng(), chain)[1]))

function loss(cord, θ)
    chain2(cord, θ) .- phi(cord, res.minimizer)
end

grid_strategy = NeuralPDE.GridTraining(0.05)
quadrature_strategy = NeuralPDE.QuadratureTraining(reltol = 1e-5, abstol = 1e-5,
                                                   maxiters = 50, batch = 100)
stochastic_strategy = NeuralPDE.StochasticTraining(400)
quasirandom_strategy = NeuralPDE.QuasiRandomTraining(400, resampling = false,
                                                     minibatch = 200)
quasirandom_strategy_resampling = NeuralPDE.QuasiRandomTraining(250)

strategies1 = [grid_strategy, quadrature_strategy]

reses_1 = map(strategies1) do strategy_
    println("Neural adapter Poisson equation, strategy: $(nameof(typeof(strategy_)))")
    prob_ = NeuralPDE.neural_adapter(loss, initθ2, pde_system, strategy_)
    res_ = Optimization.solve(prob_, ADAM(0.01); maxiters = 8000)
    prob_ = remake(prob_, u0 = res_.minimizer)
    res_ = Optimization.solve(prob_, BFGS(); maxiters = 200)
end
strategies2 = [stochastic_strategy, quasirandom_strategy]# quasirandom_strategy_resampling]
reses_2 = map(strategies2) do strategy_
    println("Neural adapter Poisson equation, strategy: $(nameof(typeof(strategy_)))")
    prob_ = NeuralPDE.neural_adapter(loss, initθ2, pde_system, strategy_)
    res_ = Optimization.solve(prob_, ADAM(0.01); maxiters = 8000)
    prob_ = remake(prob_, u0 = res_.minimizer)
    res_ = Optimization.solve(prob_, BFGS(); maxiters = 200)
end
reses_ = [reses_1; reses_2]

discretizations = map(res_ -> NeuralPDE.PhysicsInformedNN(chain2,
                                                          grid_strategy;
                                                          init_params = res_.minimizer),
                      reses_)

probs = map(discret -> NeuralPDE.discretize(pde_system, discret), discretizations)
phis = map(discret -> discret.phi, discretizations)

xs, ys = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]
analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

u_predict = reshape([first(phi([x, y], res.minimizer)) for x in xs for y in ys],
                    (length(xs), length(ys)))

u_predicts = map(zip(phis, reses_)) do (phi_, res_)
    reshape([first(phi_([x, y], res_.minimizer)) for x in xs for y in ys],
            (length(xs), length(ys)))
end

u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
                 (length(xs), length(ys)))

@test_broken u_predict≈u_real atol=1e-3
@test_broken u_predicts[1]≈u_real atol=1e-2
map(u_predicts[2:end]) do upred
    @test_broken upred≈u_real atol=1e-2
end

#using Plots
# i=3
# diff_u = abs.(u_predict .- u_real)
# diff_u_ = abs.(u_predicts[i] .- u_real)
# p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
# p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
# p5 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
# p3 = plot(xs, ys, u_predicts[i],linetype=:contourf,title = "predict_");
# p6 = plot(xs, ys, diff_u_,linetype=:contourf,title = "error_");
# plot(p2,p1,p5,p3,p6)

## Example, 2D Poisson equation, domain decomposition
println("Example, 2D Poisson equation, domain decomposition")
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

bcs = [u(0, y) ~ 0.0, u(1, y) ~ -sin(pi * 1) * sin(pi * y),
    u(x, 0) ~ 0.0, u(x, 1) ~ -sin(pi * x) * sin(pi * 1)]

# Space
x_0 = 0.0
x_end = 1.0
x_domain = Interval(x_0, x_end)
y_domain = Interval(0.0, 1.0)
domains = [x ∈ x_domain,
    y ∈ y_domain]

count_decomp = 10

# Neural network
af = Flux.tanh
inner = 12
chains = [Lux.Chain(Lux.Dense(2, inner, af), Lux.Dense(inner, inner, af),
                    Lux.Dense(inner, 1)) for _ in 1:count_decomp]
initθs = map(c->Float64.(ComponentArray(Lux.setup(Random.default_rng(), c)[1])), chains)

xs_ = infimum(x_domain):(1 / count_decomp):supremum(x_domain)
xs_domain = [(xs_[i], xs_[i + 1]) for i in 1:(length(xs_) - 1)]
domains_map = map(xs_domain) do (xs_dom)
    x_domain_ = Interval(xs_dom...)
    domains_ = [x ∈ x_domain_,
        y ∈ y_domain]
end

analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)
function create_bcs(x_domain_, phi_bound)
    x_0, x_e = x_domain_.left, x_domain_.right
    if x_0 == 0.0
        bcs = [u(0, y) ~ 0.0,
            u(x_e, y) ~ analytic_sol_func(x_e, y),
            u(x, 0) ~ 0.0,
            u(x, 1) ~ -sin(pi * x) * sin(pi * 1)]
        return bcs
    end
    bcs = [u(x_0, y) ~ phi_bound(x_0, y),
        u(x_e, y) ~ analytic_sol_func(x_e, y),
        u(x, 0) ~ 0.0,
        u(x, 1) ~ -sin(pi * x) * sin(pi * 1)]
    bcs
end

reses = []
phis = []
pde_system_map = []

for i in 1:count_decomp
    println("decomposition $i")
    domains_ = domains_map[i]
    phi_in(cord) = phis[i - 1](cord, reses[i - 1].minimizer)
    # phi_bound(x,y) = if (x isa Matrix)  phi_in(vcat(x, fill(y,size(x)))) else  phi_in(vcat(fill(x,size(y)),y)) end
    phi_bound(x, y) = phi_in(vcat(x, y))
    @register_symbolic phi_bound(x, y)
    global phi_bound
    Base.Broadcast.broadcasted(::typeof(phi_bound), x, y) = phi_bound(x, y)
    bcs_ = create_bcs(domains_[1].domain, phi_bound)
    @named pde_system_ = PDESystem(eq, bcs_, domains_, [x, y], [u(x, y)])
    push!(pde_system_map, pde_system_)
    strategy = NeuralPDE.GridTraining([0.1 / count_decomp, 0.1])

    discretization = NeuralPDE.PhysicsInformedNN(chains[i], strategy;
                                                 init_params = initθs[i])

    prob = NeuralPDE.discretize(pde_system_, discretization)
    symprob = NeuralPDE.symbolic_discretize(pde_system_, discretization)
    res_ = Optimization.solve(prob, BFGS(), maxiters = 1500)
    @show res_.minimum
    phi = discretization.phi
    push!(reses, res_)
    push!(phis, phi)
end

# function plot_(i)
#     xs, ys = [infimum(d.domain):dx:supremum(d.domain) for (dx,d) in zip([0.001,0.01], domains_map[i])]
#     u_predict = reshape([first(phis[i]([x,y],reses[i].minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
#     u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
#     diff_u = abs.(u_predict .- u_real)
#     p1 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
#     p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict");
#     p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "error");
#     plot(p1,p2,p3)
# end
# ps =[plot_(i) for i in 1:count_decomp]

function compose_result(dx)
    u_predict_array = Float64[]
    diff_u_array = Float64[]
    ys = infimum(domains[2].domain):dx:supremum(domains[2].domain)
    xs_ = infimum(x_domain):dx:supremum(x_domain)
    xs = collect(xs_)
    function index_of_interval(x_)
        for (i, x_domain) in enumerate(xs_domain)
            if x_ <= x_domain[2] && x_ >= x_domain[1]
                return i
            end
        end
    end
    for x_ in xs
        i = index_of_interval(x_)
        u_predict_sub = [first(phis[i]([x_, y], reses[i].minimizer)) for y in ys]
        u_real_sub = [analytic_sol_func(x_, y) for y in ys]
        diff_u_sub = u_predict_sub .- u_real_sub
        append!(u_predict_array, u_predict_sub)
        append!(diff_u_array, diff_u_sub)
    end
    xs, ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
    u_predict = reshape(u_predict_array, (length(xs), length(ys)))
    diff_u = reshape(diff_u_array, (length(xs), length(ys)))
    u_predict, diff_u
end
dx = 0.01
u_predict, diff_u = compose_result(dx)

inner_ = 18
af = Flux.tanh
chain2 = Lux.Chain(Lux.Dense(2, inner_, af),
                   Lux.Dense(inner_, inner_, af),
                   Lux.Dense(inner_, inner_, af),
                   Lux.Dense(inner_, inner_, af),
                   Lux.Dense(inner_, 1))

initθ2 = Float64.(ComponentArray(Lux.setup(Random.default_rng(), chain)[1]))

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

losses = map(1:count_decomp) do i
    loss(cord, θ) = chain2(cord, θ) .- phis[i](cord, reses[i].minimizer)
end

prob_ = NeuralPDE.neural_adapter(losses, initθ2, pde_system_map,
                                 NeuralPDE.GridTraining([0.1 / count_decomp, 0.1]))
res_ = Optimization.solve(prob_, BFGS(); maxiters = 2000)
@show res_.minimum
prob_ = NeuralPDE.neural_adapter(losses, res_.minimizer, pde_system_map,
                                 NeuralPDE.GridTraining(0.01))
res_ = Optimization.solve(prob_, BFGS(); maxiters = 1000)
@show res_.minimum

phi_ = NeuralPDE.Phi(chain2)

xs, ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
u_predict_ = reshape([first(phi_([x, y], res_.minimizer)) for x in xs for y in ys],
                     (length(xs), length(ys)))
u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
                 (length(xs), length(ys)))
diff_u_ = u_predict_ .- u_real

@test u_predict≈u_real rtol=0.1
@test u_predict_≈u_real rtol=0.1

# p1 = plot(xs, ys, u_predict, linetype=:contourf,title = "predict 1");
# p2 = plot(xs, ys, u_predict_,linetype=:contourf,title = "predict 2");
# p3 = plot(xs, ys, u_real, linetype=:contourf,title = "analytic");
# p4 = plot(xs, ys, diff_u,linetype=:contourf,title = "error 1");
# p5 = plot(xs, ys, diff_u_,linetype=:contourf,title = "error 2");
# plot(p1,p2,p3,p4,p5)
