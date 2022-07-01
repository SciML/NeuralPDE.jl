using NeuralPDE, Test
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using QuasiMonteCarlo, Random
import ModelingToolkit: Interval, infimum, supremum
using DomainSets
using Integrals, IntegralsCubature
using OrdinaryDiffEq, ComponentArrays
import Lux
using ComponentArrays

## Example 7, Fokker-Planck equation
println("Example 7, Fokker-Planck equation")
# the example took from this article https://arxiv.org/abs/1910.10503
@parameters x
@variables p(..)
Dx = Differential(x)
Dxx = Differential(x)^2

#2D PDE
α = 0.3
β = 0.5
_σ = 0.5
# Discretization
dx = 0.01
# here we use normalization condition: dx*p(x) ~ 1, in order to get non-zero solution.
#(α - 3*β*x^2)*p(x) + (α*x - β*x^3)*Dx(p(x)) ~ (_σ^2/2)*Dxx(p(x))
eq = [Dx((α * x - β * x^3) * p(x)) ~ (_σ^2 / 2) * Dxx(p(x))]
x_0 = -2.2
x_end = 2.2
# Initial and boundary conditions
bcs = [p(x_0) ~ 0.0, p(x_end) ~ 0.0]

# Space and time domains
domains = [x ∈ Interval(-2.2, 2.2)]

# Neural network
inn = 18
chain = Lux.Chain(Lux.Dense(1, inn, Lux.σ),
                  Lux.Dense(inn, inn, Lux.σ),
                  Lux.Dense(inn, inn, Lux.σ),
                  Lux.Dense(inn, 1))
initθ = Float64.(ComponentArray(Lux.setup(Random.default_rng(), chain)[1]))

lb = [x_0]
ub = [x_end]
function norm_loss_function(phi, θ, p)
    function inner_f(x, θ)
        dx * phi(x, θ) .- 1
    end
    prob = IntegralProblem(inner_f, lb, ub, θ)
    norm2 = solve(prob, HCubatureJL(), reltol = 1e-8, abstol = 1e-8, maxiters = 10)
    abs(norm2[1])
end
# norm_loss_function(phi,initθ,nothing)

discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             NeuralPDE.GridTraining(dx);
                                             init_params = initθ,
                                             additional_loss = norm_loss_function)

@named pde_system = PDESystem(eq, bcs, domains, [x], [p(x)])
prob = NeuralPDE.discretize(pde_system, discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

phi = discretization.phi

cb_ = function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    println("additional_loss: ", norm_loss_function(phi, p, nothing))
    return false
end

res = Optimization.solve(prob, OptimizationOptimJL.LBFGS(), maxiters = 400, callback = cb_)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 2000, callback = cb_)

C = 142.88418699042
analytic_sol_func(x) = C * exp((1 / (2 * _σ^2)) * (2 * α * x^2 - β * x^4))
xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]
u_real = [analytic_sol_func(x) for x in xs]
u_predict = [first(phi(x, res.u)) for x in xs]

@test u_predict≈u_real rtol=1e-3

### No init_params

discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             NeuralPDE.GridTraining(dx);
                                             additional_loss = norm_loss_function)

@named pde_system = PDESystem(eq, bcs, domains, [x], [p(x)])
prob = NeuralPDE.discretize(pde_system, discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

phi = discretization.phi

cb_ = function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    println("additional_loss: ", norm_loss_function(phi, p, nothing))
    return false
end

res = Optimization.solve(prob, OptimizationOptimJL.LBFGS(), maxiters = 400, callback = cb_)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 2000, callback = cb_)

C = 142.88418699042
analytic_sol_func(x) = C * exp((1 / (2 * _σ^2)) * (2 * α * x^2 - β * x^4))
xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains][1]
u_real = [analytic_sol_func(x) for x in xs]
u_predict = [first(phi(x, res.u)) for x in xs]

@test u_predict≈u_real rtol=1e-3

# plot(xs ,u_real, label = "analytic")
# plot!(xs ,u_predict, label = "predict")

## Example 8, Lorenz System (Parameter Estimation)
println("Example 8, Lorenz System")

Random.seed!(1234)
@parameters t, σ_, β, ρ
@variables x(..), y(..), z(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ_ * (y(t) - x(t)),
    Dt(y(t)) ~ x(t) * (ρ - z(t)) - y(t),
    Dt(z(t)) ~ x(t) * y(t) - β * z(t)]

bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ Interval(0.0, 1.0)]
dt = 0.05

input_ = length(domains)
n = 12
chain = [Lux.Chain(Lux.Dense(input_, n, Lux.tanh), Lux.Dense(n, n, Lux.σ),
                   Lux.Dense(n, 1)) for _ in 1:3]
#Generate Data
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 1.0)
prob = ODEProblem(lorenz!, u0, tspan)
sol = solve(prob, Tsit5(), dt = 0.1)
ts = [infimum(d.domain):dt:supremum(d.domain) for d in domains][1]

function getData(sol)
    data = []
    us = hcat(sol(ts).u...)
    ts_ = hcat(sol(ts).t...)
    return [us, ts_]
end

data = getData(sol)

#Additional Loss Function
initθs = [Float64.(ComponentArray(Lux.setup(Random.default_rng(), chain[i])[1])) for i in 1:3]
names = ntuple(i -> Symbol("θ",i), length(initθs))
flat_initθ = ComponentArray(NamedTuple{names}(i for i in initθs))

acum = [0; accumulate(+, length.(initθs))]
sep = [(acum[i] + 1):acum[i + 1] for i in 1:(length(acum) - 1)]
(u_, t_) = data
len = length(data[2])

function additional_loss(phi, θ, p)
    return sum(sum(abs2, phi[i](t_, getproperty(θ,Symbol("θ",i))) .- u_[[i], :]) / len for i in 1:1:3)
end

discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             NeuralPDE.GridTraining(dt);
                                             init_params = flat_initθ,
                                             param_estim = true,
                                             additional_loss = additional_loss)

additional_loss(discretization.phi, flat_initθ, nothing)

@named pde_system = PDESystem(eqs, bcs, domains,
                              [t], [x(t), y(t), z(t)], [σ_, ρ, β],
                              defaults = Dict([p => 1.0 for p in [σ_, ρ, β]]))
prob = NeuralPDE.discretize(pde_system, discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
sym_prob.loss_functions.full_loss_function(ComponentArray(θ=flat_initθ,p=ones(3)), Float64[])

res = Optimization.solve(prob, Optim.OptimizationOptimJL.BFGS(); maxiters = 6000)
p_ = res.minimizer[(end - 2):end]
@test sum(abs2, p_[1] - 10.00) < 0.1
@test sum(abs2, p_[2] - 28.00) < 0.1
@test sum(abs2, p_[3] - (8 / 3)) < 0.1

### No init_params

discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             NeuralPDE.GridTraining(dt);
                                             param_estim = true,
                                             additional_loss = additional_loss)

additional_loss(discretization.phi, flat_initθ, nothing)

@named pde_system = PDESystem(eqs, bcs, domains,
                              [t], [x(t), y(t), z(t)], [σ_, ρ, β],
                              defaults = Dict([p => 1.0 for p in [σ_, ρ, β]]))
prob = NeuralPDE.discretize(pde_system, discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
sym_prob.loss_functions.full_loss_function(sym_prob.flat_initθ, nothing)

res = Optimization.solve(prob, Optim.OptimizationOptimJL.BFGS(); maxiters = 6000)
p_ = res.minimizer[(end - 2):end]
@test sum(abs2, p_[1] - 10.00) < 0.1
@test sum(abs2, p_[2] - 28.00) < 0.1
@test sum(abs2, p_[3] - (8 / 3)) < 0.1

#Plotting the system
# initθ = discretization.init_params
# acum =  [0;accumulate(+, length.(initθ))]
# sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
# minimizers = [res.minimizer[s] for s in sep]
# ts = [infimum(d.domain):dt/10:supremum(d.domain) for d in domains][1]
# u_predict  = [[discretization.phi[i]([t],minimizers[i])[1] for t in ts] for i in 1:3]
# plot(sol)
# plot!(ts, u_predict, label = ["x(t)" "y(t)" "z(t)"])

## approximation from data
println("Approximation of function from data and additional_loss")

@parameters x
@variables u(..)
eq = [u(0) ~ u(0)]
bc = [u(0) ~ u(0)]
x0 = 0
x_end = pi
dx = pi / 10
domain = [x ∈ Interval(x0, x_end)]

hidden = 10
chain = Lux.Chain(Lux.Dense(1, hidden, Lux.tanh),
                  Lux.Dense(hidden, hidden, Lux.sin),
                  Lux.Dense(hidden, hidden, Lux.tanh),
                  Lux.Dense(hidden, 1))

strategy = NeuralPDE.GridTraining(dx)
xs = collect(x0:dx:x_end)'
aproxf_(x) = @. cos(pi * x)
data = aproxf_(xs)

function additional_loss_(phi, θ, p)
    sum(abs2, phi(xs, θ) .- data)
end

discretization = NeuralPDE.PhysicsInformedNN(chain, strategy;
                                             additional_loss = additional_loss_)

phi = discretization.phi
phi(xs, initθ)
additional_loss_(phi, initθ, nothing)

@named pde_system = PDESystem(eq, bc, domain, [x], [u(x)])
prob = NeuralPDE.discretize(pde_system, discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01), maxiters = 500)
prob = remake(prob, u0 = res.minimizer)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(), maxiters = 500)

@test phi(xs, res.u)≈aproxf_(xs) rtol=0.01

# xs_ = xs'
# plot(xs_,data')
# plot!(xs_, phi(xs,res.u)')

# func(x,y) = -20.0 * exp(-0.2 * sqrt(0.5 * (x^2 + y^2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
# func(x,y) = -abs(sin(x) * cos(y) * exp(abs(1 - (sqrt(x^2 + y^2)/pi))))
