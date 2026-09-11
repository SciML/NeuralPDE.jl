using ModelingToolkit, ModelingToolkitNeuralNets, Optimization, OptimizationOptimJL, Symbolics,
      Lux, NNlib, StableRNGs, Random, Plots, LineSearches, ComponentArrays

@variables x y
Dx = Differential(x)
Dy = Differential(y)
Dxx = Dx^2
Dyy = Dy^2

# Symbolic Neural Network
chain = Lux.Chain(
    Lux.Dense(2, 16, σ),
    Lux.Dense(16, 16, σ),
    Lux.Dense(16, 1)
)
NN_expr, p_nn_sym = SymbolicNeuralNetwork(; chain=chain, n_input=2, n_output=1, rng=StableRNG(42))
u_expr = NN_expr([x, y], p_nn_sym)[1]


# PDE and Boundary Conditions
f_rhs = sin(π * x) * sin(π * y)
pde_residual_expr = expand_derivatives(Dxx(u_expr) + Dyy(u_expr)) - f_rhs

bc_x0_residual = substitute(u_expr, x => 0.0)
bc_x1_residual = substitute(u_expr, x => 1.0)
bc_y0_residual = substitute(u_expr, y => 0.0)
bc_y1_residual = substitute(u_expr, y => 1.0)


# Data Sampling
num_col = 200
num_bc = 50
Random.seed!(123)

collocation_pts_x = rand(num_col)
collocation_pts_y = rand(num_col)

bc_x0_y_data = rand(num_bc)
bc_x1_y_data = rand(num_bc)
bc_y0_x_data = rand(num_bc)
bc_y1_x_data = rand(num_bc)


@parameters col_x[1:num_col] col_y[1:num_col]
@parameters bcx0_y_sym[1:num_bc] bcx1_y_sym[1:num_bc]
@parameters bcy0_x_sym[1:num_bc] bcy1_x_sym[1:num_bc]

bc_x0_residual^2
substitute(bc_x0_residual^2, Dict(y =>0.0))

# Loss
pde_loss_expr = sum(Symbolics.fast_substitute(pde_residual_expr^2, Dict(x => col_x[i], y => col_y[i])) for i in 1:num_col)

bc_loss_expr = sum(
    Symbolics.fast_substitute(bc_x0_residual^2, Dict(y => bcx0_y_sym[i])) +
    Symbolics.fast_substitute(bc_x1_residual^2, Dict(y => bcx1_y_sym[i])) +
    Symbolics.fast_substitute(bc_y0_residual^2, Dict(x => bcy0_x_sym[i])) +
    Symbolics.fast_substitute(bc_y1_residual^2, Dict(x => bcy1_x_sym[i]))
    for i in 1:num_bc
)

total_loss = pde_loss_expr + bc_loss_expr

# Optimization
sym_data = vcat(col_x, col_y, bcx0_y_sym, bcx1_y_sym, bcy0_x_sym, bcy1_x_sym)
u0 = randn(Float64, length(p_nn_sym))
rng = Random.GLOBAL_RNG
init_params = Lux.initialparameters(rng, chain)
ca = ComponentArray(init_params)

flat_data = vcat(
    collocation_pts_x,
    collocation_pts_y,
    bc_x0_y_data,
    bc_x1_y_data,
    bc_y0_x_data,
    bc_y1_x_data
)

@assert length(sym_data) == length(flat_data)

@named system = OptimizationSystem(total_loss, p_nn_sym, [sym_data; NN_expr])
system = complete(system)

prob = OptimizationProblem(system, [p_nn_sym => u0], [sym_data .=> flat_data; NN_expr => ModelingToolkitNeuralNets.StatelessApplyWrapper(chain, typeof(ca))]; grad=true)
result = solve(prob, LBFGS(linesearch=LineSearches.BackTracking()); maxiters=1000)

# Evaluation
function u_pred(xval, yval, chain, optimized_params)
    ps = convert(typeof(ca),optimized_params)
    return ModelingToolkitNeuralNets.stateless_apply(chain, [xval, yval], ps)[1]
end

u_exact(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

xs = range(0, 1; length=100)
ys = range(0, 1; length=100)

u_approx = [u_pred(x, y, chain, result.minimizer) for y in ys, x in xs]
u_truth  = [u_exact(x, y) for y in ys, x in xs]
u_error  = abs.(u_approx .- u_truth)

# Plots
layout = @layout [a b c]
plt1 = heatmap(xs, ys, u_approx; title="Predicted u(x,y)", xlabel="x", ylabel="y", colorbar_title="u")
plt2 = heatmap(xs, ys, u_truth;  title="Exact u(x,y)",     xlabel="x", ylabel="y", colorbar_title="u")
plt3 = heatmap(xs, ys, u_error;  title="|u_pred - u_exact|", xlabel="x", ylabel="y", colorbar_title="abs error")

plot(plt1, plt2, plt3; layout=layout, size=(1100, 300))
