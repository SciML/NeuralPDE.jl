### 468

using NeuralPDE, Lux, OptimizationOptimisers
import ModelingToolkit: Interval, infimum, supremum
using Random
using LuxCUDA
using ComponentArrays

gpud = gpu_device()

rng = Random.default_rng()

@parameters t
@variables x(..), y(..)

Dt = Differential(t)
Dtt = Differential(t)^2

# 2D PDE
M = 60.0 # mass of the planet
X0, Y0 = 0.0, 0.0 #position of the planet

eq = [Dtt(x(t)) ~ -M / ((x(t) - X0)^2 + (y(t) - Y0)^2)^1.5 * (x(t) - X0),
    Dtt(y(t)) ~ -M / ((x(t) - X0)^2 + (y(t) - Y0)^2)^1.5 * (y(t) - Y0)]

# Initial and boundary conditions
x0, y0 = 10.0, 10.0 # position of the test particle
vx0, vy0 = 1.0, -5.0  #initial velocity of the test particle
bcs = [x(0.0) ~ x0,
    y(0.0) ~ y0,
    Dt(x(0.0)) ~ vx0,
    Dt(y(0.0)) ~ vy0]

# Space and time domains
domains = [t ∈ Interval(0.0, 15.0)]

inner = 25
chain = [Chain(Dense(1, inner, Lux.σ),
             Dense(inner, inner, Lux.σ),
             Dense(inner, inner, Lux.σ),
             Dense(inner, inner, Lux.σ),
             Dense(inner, 1)) for _ in 1:2]

ps1, _ = Lux.setup(rng, chain[1])
ps2, _ = Lux.setup(rng, chain[2])
ps1 = ps1 |> ComponentArray |> gpud .|> Float64
ps2 = ps2 |> ComponentArray |> gpud .|> Float64

strategy = GridTraining(0.01)
discretization = PhysicsInformedNN(chain,
    strategy;
    init_params = [ps1, ps2])

@named pde_system = PDESystem(eq, bcs, domains, [t], [x(t), y(t)])
prob = discretize(pde_system, discretization)
symprob = symbolic_discretize(pde_system, discretization)

pde_inner_loss_functions = symprob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = symprob.loss_functions.bc_loss_functions

cb = function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
    return false
end

opt = OptimizationOptimisers.Adam(0.1)
res = solve(prob, opt; callback = cb, maxiters = 10)

##### 
using KernelAbstractions, StaticArrays, LuxCUDA
@kernel function custom_broadcast!(f, du, @Const(out), @Const(p), @Const(t))
    i = @index(Global, Linear)
    @views @inbounds x = f(out[:, i], p, t[i])
    du[:, i] .= x
end

function linear2(u, p, t)
    return SVector{2}(p[1] * cos(2pi * t), p[2] * sin(2pi * t))
end

out = CuArray(rand(2, 100))
p = CuArray([1.0, 1.0])
t = CuArray(rand(100))
du = similar(out)
k = custom_broadcast!(CUDABackend())
# KernelAbstractions.workgroupsize(CUDABackend(size(out, 2)))
k(linear2, du, out, p, t; workgroupsize = 64, ndrange = 100)

k2 = custom_broadcast!(CPU(), 32)
out = rand(2, 100)
p = [1.0, 1.0]
t = rand(100)
du = similar(out)
k2(linear, du, out, p, t; ndrange = 100)
