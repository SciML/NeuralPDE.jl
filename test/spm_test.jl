begin
using LinearAlgebra
#using PyCall
using Flux
using DiffEqFlux
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
using GalacticOptim
using Optim
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using SciMLBase
using DelimitedFiles
using CSV
using Plots
using Random
using TensorBoardLogger
end

#function main()
begin
Random.seed!(100)
# ('negative particle',) -> rn
# ('positive particle',) -> rp
@parameters t rn rp
# 'Discharge capacity [A.h]' -> Q
# 'X-averaged negative particle concentration' -> c_s_n_xav
# 'X-averaged positive particle concentration' -> c_s_p_xav
@variables Q(..) c_s_n_xav(..) c_s_p_xav(..)
Dt = Differential(t)
Drn = Differential(rn)
Drp = Differential(rp)

eqs = [
Dt(Q(t)) ~ 4.27249308415467,
# 'X-averaged negative particle concentration' equation
Dt(c_s_n_xav(t, rn)) ~ 8.813457647415216 * (Drn(Drn(c_s_n_xav(t, rn))) + 2 / rn * Drn(c_s_n_xav(t, rn))),
# 'X-averaged positive particle concentration' equation
Dt(c_s_p_xav(t, rp)) ~ 22.598609352346717 * (Drp(Drp(c_s_p_xav(t, rp))) + 2 / rp * Drp(c_s_p_xav(t, rp))),
]

ics_bcs = [
Q(0) ~ 0.0,
c_s_n_xav(0, rn) ~ 0.8,
c_s_p_xav(0, rp) ~ 0.6,
Drn(c_s_n_xav(t, 0.01)) ~ 0.0,
Drn(c_s_n_xav(t, 1.0)) ~ -0.14182855923368468,
Drp(c_s_p_xav(t, 0.01)) ~ 0.0,
Drp(c_s_p_xav(t, 1.0)) ~ 0.03237700710041634,
]

t_domain = IntervalDomain(0.0, 0.15) 
rn_domain = IntervalDomain(0.01, 1.0)
rp_domain = IntervalDomain(0.01, 1.0)

domains = [
t in t_domain,
rn in rn_domain,
rp in rp_domain,
]
ind_vars = [t, rn, rp]
dep_vars = [Q(t), c_s_n_xav(t, rn), c_s_p_xav(t, rp)]

@named SPM_pde_system = PDESystem(eqs, ics_bcs, domains, ind_vars, dep_vars)

run_number = 2

loggerloc = joinpath("logs", "SPM_pde_system", "$run_number")

if !isdir(loggerloc)
    mkdir(loggerloc)
end

logger = TBLogger(loggerloc, tb_append)

num_dim = 50
nonlin = Flux.gelu
strategy = NeuralPDE.QuadratureTraining(;quadrature_alg=CubatureJLh(),abstol=1e-4, reltol=1, maxiters=2000, batch=128)
strategy = NeuralPDE.StochasticTraining(256)
in_dims = [1, 2, 2]
num_hid = 2
chains_ = [FastChain(FastDense(in_dim,num_dim,nonlin),
                    [FastDense(num_dim,num_dim,nonlin) for i in 1:num_hid]...,
                    FastDense(num_dim,1)) for in_dim in in_dims]
adaloss = NeuralPDE.MiniMaxAdaptiveLoss(20; pde_max_optimiser=ADAM(1e-1), bc_max_optimiser=ADAM(1e1), pde_loss_weights=1e-3, bc_loss_weights=1e3)
iteration_count_arr = [0]
discretization = NeuralPDE.PhysicsInformedNN(chains_, strategy; adaptive_loss=adaloss, iteration=iteration_count_arr, logger=logger)
prob = NeuralPDE.discretize(SPM_pde_system,discretization)

initθ = vcat(discretization.init_params...)
opt = ADAM(3e-4)
maxiters = 50_000

most_recent_params = [initθ]

function cb(p,l)
    iteration_count = iteration_count_arr[1]
    println("Current loss is: $l, iteration is: $(iteration_count)")
    iteration_count_arr[1] += 1
    if iteration_count % 100 == 0
        most_recent_params[1] = copy(p)
    end
    return false
end
end

try
    res = GalacticOptim.solve(prob, opt; cb = cb, maxiters=50_000)
catch e
    println("Error: $e")
    most_recent_params
end

#return res
#end


