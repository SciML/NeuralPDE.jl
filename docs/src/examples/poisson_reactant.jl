using Lux, Reactant, Random, Statistics, Enzyme, MLUtils, ConcreteStructs, Printf, Optimisers, CairoMakie, LinearAlgebra, OnlineStats, LossFunctions

f_vec(xy) = -sin.(π .* xy[1, :]) .* sin.(π .* xy[2, :])

struct PINN_Poisson{U,P,Q} <: AbstractLuxContainerLayer{(:u, :p, :q)}
    u::U
    p::P
    q::Q
end

function create_mlp(act, hidden_dims, in_dims=2)
    return Chain(
        Dense(in_dims => hidden_dims, act),
        Dense(hidden_dims => hidden_dims, act),
        Dense(hidden_dims => hidden_dims, act),
        Dense(hidden_dims => 1),
    )
end

function PINN_Poisson(; hidden_dims::Int=128)
    return PINN_Poisson(
        create_mlp(Lux.swish, hidden_dims),
        create_mlp(Lux.swish, hidden_dims),
        create_mlp(Lux.swish, hidden_dims),
    )
end

function (pinn::PINN_Poisson)(xy, ps, st)
    u, st_u = Lux.apply(pinn.u, xy, ps.u, st.u)
    p, st_p = Lux.apply(pinn.p, xy, ps.p, st.p)
    q, st_q = Lux.apply(pinn.q, xy, ps.q, st.q)
    return (u=u, p=p, q=q), merge(st, (u=st_u, p=st_p, q=st_q))
end

rng = Random.default_rng()
Random.seed!(rng, 0)

pinn_poisson = PINN_Poisson()
ps_poisson, st_poisson = Lux.setup(rng, pinn_poisson) |> reactant_device()

analytical_solution_poisson(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)
analytical_solution_poisson(xy_batch) = analytical_solution_poisson.(xy_batch[1, :], xy_batch[2, :])

grid_len = 64
grid_x = range(0.0f0, 1.0f0; length=grid_len)
grid_y = range(0.0f0, 1.0f0; length=grid_len)
xy_pde_pts = stack([[elem...] for elem in vec(collect(Iterators.product(grid_x, grid_y)))])

target_data_values = analytical_solution_poisson(xy_pde_pts)
target_data_values = reshape(target_data_values, 1, :)

bc_len = 256
x_bc_scalar = collect(range(0.0f0, 1.0f0; length=bc_len))
y_bc_scalar = collect(range(0.0f0, 1.0f0; length=bc_len))

xy_bc_points = hcat(
    [zeros(Float32, bc_len)'; y_bc_scalar'],
    [ones(Float32, bc_len)'; y_bc_scalar'],
    [x_bc_scalar'; zeros(Float32, bc_len)'],
    [x_bc_scalar'; ones(Float32, bc_len)']
)

target_bc_values = zeros(Float32, size(xy_bc_points, 2))
target_bc_values = reshape(target_bc_values, 1, :)

min_target_data, max_target_data = extrema(target_data_values)
min_target_bc, max_target_bc = extrema(target_bc_values)
global_min_output_val = min(min_target_data, min_target_bc)
global_max_output_val = max(max_target_data, max_target_bc)

min_xy_pde = minimum(xy_pde_pts)
max_xy_pde = maximum(xy_pde_pts)
xy_pde_pts_normalized = (xy_pde_pts .- min_xy_pde) ./ (max_xy_pde - min_xy_pde)

min_xy_bc = minimum(xy_bc_points)
max_xy_bc = maximum(xy_bc_points)
xy_bc_points_normalized = (xy_bc_points .- min_xy_bc) ./ (max_xy_bc - min_xy_bc)

target_data_values_normalized = (target_data_values .- global_min_output_val) ./ (global_max_output_val - global_min_output_val)
target_bc_values_normalized = (target_bc_values .- global_min_output_val) ./ (global_max_output_val - global_min_output_val)

xs_plot = 0.0f0:0.02f0:1.0f0
ys_plot = 0.0f0:0.02f0:1.0f0
grid_plot_x = collect(xs_plot)
grid_plot_y = collect(ys_plot)

u_real_plot = [analytical_solution_poisson(x, y) for x in xs_plot, y in ys_plot]

fig_true = Figure()
ax_true = CairoMakie.Axis(fig_true[1, 1]; xlabel="x", ylabel="y", title="True Analytical Solution")
CairoMakie.heatmap!(ax_true, xs_plot, ys_plot, u_real_plot)
CairoMakie.contour!(ax_true, xs_plot, ys_plot, u_real_plot; levels=10, linewidth=2, color=:black)
Colorbar(fig_true[1, 2]; limits=extrema(u_real_plot), label="True u")
display(fig_true)

@views function physics_loss_function(u_net::StatefulLuxLayer, p_net::StatefulLuxLayer, q_net::StatefulLuxLayer, xy_batch_normalized::AbstractArray)
    dpdx_norm = Enzyme.gradient(Enzyme.Reverse, (x) -> sum(p_net(x)), xy_batch_normalized)[1][1:1, :]
    dqdy_norm = Enzyme.gradient(Enzyme.Reverse, (x) -> sum(q_net(x)), xy_batch_normalized)[1][2:2, :]
    
    xy_batch_actual = xy_batch_normalized .* (max_xy_pde .- min_xy_pde) .+ min_xy_pde
    f_vals_actual = f_vec(xy_batch_actual)
    physics_scale_factor = 1.0f0 / (global_max_output_val - global_min_output_val)
    f_vals_scaled_for_physics_loss = f_vals_actual .* physics_scale_factor

    pde_residual_component = dpdx_norm .+ dqdy_norm .- f_vals_scaled_for_physics_loss
    pde_res_loss = mean(abs2, pde_residual_component)

    ∂u_∂xy_norm = Enzyme.gradient(Enzyme.Reverse, (x) -> sum(u_net(x)), xy_batch_normalized)[1]
    ∂u_∂x_norm = ∂u_∂xy_norm[1:1, :]
    ∂u_∂y_norm = ∂u_∂xy_norm[2:2, :]

    p_pred_norm = p_net(xy_batch_normalized)
    q_pred_norm = q_net(xy_batch_normalized)

    p_consistency_residual = p_pred_norm .- ∂u_∂x_norm
    q_consistency_residual = q_pred_norm .- ∂u_∂y_norm
    consistency_res_loss = mean(abs2, p_consistency_residual) + mean(abs2, q_consistency_residual)

    total_physics_loss = pde_res_loss + consistency_res_loss

    mean_abs_f_val = mean(abs.(f_vals_scaled_for_physics_loss))
    mean_abs_laplacian_val = mean(abs.(dpdx_norm .+ dqdy_norm))

    return total_physics_loss, mean_abs_f_val, mean_abs_laplacian_val
end

function mse_loss_function(u_net::StatefulLuxLayer, target_normalized::AbstractArray, xy_normalized::AbstractArray)
    u_pred_normalized = u_net(xy_normalized)
    return MSELoss()(u_pred_normalized, target_normalized)
end

function loss_function(model, ps, st, (xy_pde_normalized, target_data_normalized, xy_bc_normalized, target_bc_normalized))
    u_net = StatefulLuxLayer{true}(model.u, ps.u, st.u)
    p_net = StatefulLuxLayer{true}(model.p, ps.p, st.p)
    q_net = StatefulLuxLayer{true}(model.q, ps.q, st.q)

    physics_loss, mean_abs_f, mean_abs_dpdx_dqdy = physics_loss_function(u_net, p_net, q_net, xy_pde_normalized)
    
    data_loss = mse_loss_function(u_net, target_data_normalized, xy_pde_normalized)
    bc_loss = mse_loss_function(u_net, target_bc_normalized, xy_bc_normalized)

    w_physics = 1.0f0
    w_data = 1000.0f0
    w_bc = 5000.0f0

    total_loss = w_physics * physics_loss + w_data * data_loss + w_bc * bc_loss

    updated_st_u = u_net.st
    updated_st_p = p_net.st
    updated_st_q = q_net.st
    updated_st_overall = merge(st, (u=updated_st_u, p=updated_st_p, q=updated_st_q))

    return (
        total_loss,
        updated_st_overall,
        (; physics_loss, data_loss, bc_loss, mean_abs_f, mean_abs_dpdx_dqdy)
    )
end

train_state = Lux.Training.TrainState(pinn_poisson, ps_poisson, st_poisson, Adam(0.001f0))

lr = i -> i < 10000 ? 0.001f0 : (i < 30000 ? 0.0001f0 : 0.00001f0)

bc_dataloader = MLUtils.DataLoader(
    (xy_bc_points_normalized, target_bc_values_normalized); batchsize=128, shuffle=true, partial=false
) |> reactant_device()
pde_dataloader = MLUtils.DataLoader(
    (xy_pde_pts_normalized, target_data_values_normalized); batchsize=128, shuffle=true, partial=false
) |> reactant_device();

total_loss_tracker, physics_loss_tracker, data_loss_tracker, bc_loss_tracker, mean_abs_f_tracker, mean_abs_dpdx_dqdy_tracker = ntuple(
    _ -> OnlineStats.CircBuff(Float32, 32; rev=true), 6
) 

iter = 1
maxiters = 50000 

for ((xy_pde_batch, target_data_batch), (xy_bc_batch, target_bc_batch)) in
    zip(Iterators.cycle(pde_dataloader), Iterators.cycle(bc_dataloader))

    data_tuple = (xy_pde_batch, target_data_batch, xy_bc_batch, target_bc_batch)

    Optimisers.adjust!(train_state, lr(iter))

    _, loss, stats, ts_new = Lux.Training.single_train_step!(
        AutoEnzyme(),
        loss_function,
        data_tuple,
        train_state;
        return_gradients=Val(false),
    )
    
    train_state = ts_new

    fit!(total_loss_tracker, Float32(loss))
    fit!(physics_loss_tracker, Float32(stats.physics_loss))
    fit!(data_loss_tracker, Float32(stats.data_loss))
    fit!(bc_loss_tracker, Float32(stats.bc_loss))
    fit!(mean_abs_f_tracker, Float32(stats.mean_abs_f)) 
    fit!(mean_abs_dpdx_dqdy_tracker, Float32(stats.mean_abs_dpdx_dqdy))

    mean_loss = mean(OnlineStats.value(total_loss_tracker))
    mean_physics_loss = mean(OnlineStats.value(physics_loss_tracker))
    mean_data_loss = mean(OnlineStats.value(data_loss_tracker))
    mean_bc_loss = mean(OnlineStats.value(bc_loss_tracker))
    mean_mean_abs_f = mean(OnlineStats.value(mean_abs_f_tracker)) 
    mean_mean_abs_dpdx_dqdy = mean(OnlineStats.value(mean_abs_dpdx_dqdy_tracker))

    isnan(loss) && throw(ArgumentError("NaN Loss Detected"))

    if iter % 1000 == 1 || iter == maxiters
        @printf "Iteration: [%6d/%6d] \t Loss: %.9f (%.9f) \t Physics Loss: %.9f (%.9f) \t Data Loss: %.9f (%.9f) \t BC Loss: %.9f (%.9f) \t |f|: %.3f (|dp/dx+dq/dy|): %.3f\n" iter maxiters loss mean_loss stats.physics_loss mean_physics_loss stats.data_loss mean_data_loss stats.bc_loss mean_bc_loss stats.mean_abs_f mean_mean_abs_dpdx_dqdy
    end

    iter += 1
    iter ≥ maxiters && break
end

cdev = cpu_device()
trained_u = StatefulLuxLayer{true}(
    pinn_poisson.u, cdev(train_state.parameters.u), cdev(train_state.states.u)
)

xs_plot_grid = collect(xs_plot)
ys_plot_grid = collect(ys_plot)
xy_plot_cpu = stack([[elem...] for elem in vec(collect(Iterators.product(xs_plot_grid, ys_plot_grid)))]) 

min_xy_plot = minimum(xy_plot_cpu)
max_xy_plot = maximum(xy_plot_cpu)
xy_plot_cpu_normalized = (xy_plot_cpu .- min_xy_plot) ./ (max_xy_plot - min_xy_plot)

u_pred_plot_normalized = trained_u(xy_plot_cpu_normalized)
u_pred_plot_normalized = reshape(u_pred_plot_normalized, length(xs_plot), length(ys_plot))
u_pred_plot = (u_pred_plot_normalized .* (global_max_output_val - global_min_output_val)) .+ global_min_output_val

fig_trained = Figure()
ax_trained = CairoMakie.Axis(fig_trained[1, 1]; xlabel="x", ylabel="y", title="Trained Solution")
CairoMakie.heatmap!(ax_trained, xs_plot, ys_plot, u_pred_plot)
CairoMakie.contour!(ax_trained, xs_plot, ys_plot, u_pred_plot; levels=10, linewidth=2, color=:black)
CairoMakie.Colorbar(fig_trained[1, 2]; limits=extrema(u_pred_plot), label="Predicted u")
display(fig_trained)

u_real_plot_final = [analytical_solution_poisson(x, y) for x in xs_plot, y in ys_plot]
abs_error_plot = abs.(u_pred_plot .- u_real_plot_final)

fig_error = Figure()
ax_error = CairoMakie.Axis(fig_error[1, 1]; xlabel="x", ylabel="y", title="Absolute Error")
CairoMakie.heatmap!(ax_error, xs_plot, ys_plot, abs_error_plot, colorrange=(0, maximum(abs_error_plot)))
CairoMakie.Colorbar(fig_error[1, 2]; limits=(0, maximum(abs_error_plot)), label="Absolute Error")
display(fig_error)

println("Maximum Absolute Error: ", maximum(abs_error_plot))