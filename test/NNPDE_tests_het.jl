

######### ZOE EXAMPLE
@parameters x y
@variables p(..) q(..) r(..) s(..)
Dx = Differential(x)
Dy = Differential(y)

 # 2D PDE
eq  = Dx(p(x)) + Dy(q(y)) + Dx(r(x, y)) + Dy(s(y, x)) + p(x) + q(y) + r(x, y) + s(y, x) ~ 0

 # Initial and boundary conditions
bcs = [p(1) ~ 0.f0, q(-1) ~ 0.0f0,
         r(x, -1) ~ 0.f0, r(1, y) ~ 0.0f0, 
         s(y, 1) ~ 0.0f0, s(-1, x) ~ 0.0f0]
 # bcs = [s(y, 1) ~ 0.0f0]
 # Space and time domains
domains = [x ∈ IntervalDomain(0.0, 1.0),
             y ∈ IntervalDomain(-1.0, 0.0)]

 # chain_ = FastChain(FastDense(2,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,1))
numhid = 3
fastchains = [[FastChain(FastDense(1, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2];
               [FastChain(FastDense(2, numhid, Flux.σ), FastDense(numhid, numhid, Flux.σ), FastDense(numhid, 1)) for i in 1:2]]
discretization = NeuralPDE.PhysicsInformedNN(fastchains,
                                                 StochasticTraining(30))

_indvars = [x,y]
_depvars = [p(x), q(y), r(x, y), s(y, x)]

depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)

pde_system = PDESystem(eq, bcs, domains, indvars, depvars)

sym_prob = NeuralPDE.symbolic_discretize(pde_system, discretization)
prob = NeuralPDE.discretize(pde_system, discretization)
initθ = discretization.init_params
initθvec = vcat(initθ...)
prob.f(initθvec, [])
res = GalacticOptim.solve(prob, ADAM(0.1); maxiters=3)
phi = discretization.phi




######### NORMAL EXAMPLE
@parameters t, x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# 2D PDE
eq  = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - (0.01 / pi) * Dxx(u(t, x)) ~ 0

# Initial and boundary conditions
bcs = [u(0, x) ~ -sin(pi * x),
       u(t, -1) ~ 0.,
       u(t, 1) ~ 0.,
       u(t, -1) ~ u(t, 1)]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(-1.0, 1.0)]
# Discretization
dx = 0.05
# Neural network
chain = FastChain(FastDense(2, 16, Flux.σ), FastDense(16, 16, Flux.σ), FastDense(16, 1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
eltypeθ = eltype(initθ)
parameterless_type_θ = DiffEqBase.parameterless_type(initθ)
strategy = NeuralPDE.GridTraining(dx)

phi = NeuralPDE.get_phi(chain, parameterless_type_θ)
derivative = NeuralPDE.get_numeric_derivative()


_indvars = [t,x]
_depvars = [u]

depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars, _depvars)

Base.Broadcast.broadcasted(::typeof(transform_expression(eq, dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)), cord, θ, phi) = transform_expression(eq, dict_indvars, dict_depvars, dict_depvar_input, chain, eltypeθ, strategy)(cord, θ, phi)

_pde_loss_function = NeuralPDE.build_loss_function(eq,indvars,depvars, dict_depvar_input,
                                                   phi,derivative,chain,initθ,strategy)

bc_indvars = NeuralPDE.get_variables(bcs, indvars, depvars)
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,dict_depvar_input,
                                                    phi,derivative,chain,initθ,strategy,
                                                    bc_indvars=bc_indvar) for (bc, bc_indvar) in zip(bcs, bc_indvars)]

train_sets = NeuralPDE.generate_training_sets(domains, dx, [eq], bcs, eltypeθ, indvars, depvars)
train_domain_set, train_bound_set = train_sets


pde_loss_function = NeuralPDE.get_loss_function(_pde_loss_function,
                                                train_domain_set[1],
                                                eltypeθ,parameterless_type_θ,
                                                strategy)

bc_loss_functions = [NeuralPDE.get_loss_function(loss,set,
                                                 eltypeθ, parameterless_type_θ,
                                                 strategy) for (loss, set) in zip(_bc_loss_functions, train_bound_set)]


loss_functions = [pde_loss_function; bc_loss_functions]
loss_function__ = θ -> sum(map(l -> l(θ), loss_functions))

function loss_function_(θ, p)
    return loss_function__(θ)
end

f = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f, initθ)

cb_ = function (p, l)
    println("loss: ", l, "losses: ", map(l -> l(p), loss_functions))
    return false
end

# optimizer
opt = BFGS()
res = GalacticOptim.solve(prob, opt; cb=cb_, maxiters=2) 

