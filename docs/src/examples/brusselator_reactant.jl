#Scalar Indexing Error
using Lux, Random, Reactant, Enzyme, MLUtils, Optimisers, OnlineStats, CairoMakie, Statistics, Printf, CUDA

const T = Float32
global device_func = reactant_device(; force=true)

struct PINNBrusselator{U,V} <: AbstractLuxContainerLayer{(:u, :v)}
    u::U
    v::V
end

function create_mlp(act, hidden)
    Chain(Dense(3 => hidden, act), Dense(hidden => hidden, act),
          Dense(hidden => hidden, act), Dense(hidden => 1))
end

function PINNBrusselator(; hidden=128)
    PINNBrusselator(create_mlp(Lux.swish, hidden), create_mlp(Lux.swish, hidden))
end

struct Normalizer{T}
    min_vals::T
    max_vals::T
end

(n::Normalizer)(x) = (x .- n.min_vals) ./ (n.max_vals .- n.min_vals)
inv(n::Normalizer) = x -> x .* (n.max_vals .- n.min_vals) .+ n.min_vals

function u₀(x, y) T(22.0) * (y * (1 - y))^(3/2) end
function v₀(x, y) T(27.0) * (x * (1 - x))^(3/2) end

f(x, y, t) = (t ≥ 1.1f0 && (x - 0.3f0)^2 + (y - 0.6f0)^2 ≤ 0.01f0) ? T(5.0) : T(0.0)

function f_batch(coords)
    x, y, t = coords[1, :], coords[2, :], coords[3, :]
    mask = ((x .- 0.3f0).^2 .+ (y .- 0.6f0).^2 .<= 0.01f0) .& (t .>= 1.1f0)
    ifelse.(mask, T(5.0), T(0.0))
end

function first_derivs(net::StatefulLuxLayer, xyt)
    grads = Enzyme.gradient(Enzyme.Reverse, sum ∘ net, xyt)[1]
    grads[1:1, :], grads[2:2, :], grads[3:3, :]
end

function laplacian(net::StatefulLuxLayer, xyt)
    fx(x) = sum(first_derivs(net, x)[1])
    fy(x) = sum(first_derivs(net, x)[2])
    d2x = Enzyme.gradient(Enzyme.Reverse, fx, xyt)[1][1:1, :]
    d2y = Enzyme.gradient(Enzyme.Reverse, fy, xyt)[1][2:2, :]
    d2x .+ d2y
end

function pde_residual(u, v, xyt, α, f_vals)
    u_pred = u(xyt)
    v_pred = v(xyt)
    _, _, ∂u_∂t = first_derivs(u, xyt)
    _, _, ∂v_∂t = first_derivs(v, xyt)
    ∇²u = laplacian(u, xyt)
    ∇²v = laplacian(v, xyt)

    res_u = ∂u_∂t .- (T(1.0) .+ u_pred.^2 .* v_pred .- T(4.4) .* u_pred .+ α .* ∇²u .+ f_vals)
    res_v = ∂v_∂t .- (T(3.4) .* u_pred .- u_pred.^2 .* v_pred .+ α .* ∇²v)
    res_u, res_v
end

function ic_loss(u, v, xyt, target_u, target_v)
    pu, _ = u(xyt)
    pv, _ = v(xyt)
    mean(abs2, pu .- target_u) + mean(abs2, pv .- target_v)
end

function bc_loss(u, v, x0, x1, y0, y1)
    ux0, _ = u(x0); ux1, _ = u(x1)
    uy0, _ = u(y0); uy1, _ = u(y1)
    vx0, _ = v(x0); vx1, _ = v(x1)
    vy0, _ = v(y0); vy1, _ = v(y1)
    mean(abs2, ux0 .- ux1) + mean(abs2, uy0 .- uy1) +
    mean(abs2, vx0 .- vx1) + mean(abs2, vy0 .- vy1)
end

function loss_fn(model, ps, st, data)
    u_net = StatefulLuxLayer{true}(model.u, ps.u, st.u)
    v_net = StatefulLuxLayer{true}(model.v, ps.v, st.v)
    pde_xyt, ic_data, bc_data, denorm, α = data

    actual = denorm(pde_xyt)
    fvals = f_batch(actual)

    res_u, res_v = pde_residual(u_net, v_net, pde_xyt, α, fvals)
    loss_pde = mean(abs2, res_u) + mean(abs2, res_v)

    ic_xyt, u_ic, v_ic = ic_data
    loss_ic = ic_loss(u_net, v_net, ic_xyt, u_ic, v_ic)

    x0, x1, y0, y1 = bc_data
    loss_bc = bc_loss(u_net, v_net, x0, x1, y0, y1)

    loss = loss_pde + 1000f0 * loss_ic + 100f0 * loss_bc
    return loss, (; u=st.u, v=st.v), (; loss_pde, loss_ic, loss_bc)
end

function train_brusselator!()
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    α = T(0.001)
    tspan = (0f0, 11.5f0)
    xspan = (0f0, 1f0)
    yspan = (0f0, 1f0)

    pde_n = 10_000; ic_n = 2000; bc_n = 2000

    x_pde = rand(rng, T, pde_n)
    y_pde = rand(rng, T, pde_n)
    t_pde = rand(rng, T, pde_n)

    xyt_pde = vcat(x_pde', y_pde', t_pde')

    x_ic = rand(rng, T, ic_n)
    y_ic = rand(rng, T, ic_n)
    t_ic = fill(T(0.0), ic_n)

    xyt_ic = vcat(x_ic', y_ic', t_ic')
    u_ic = reshape(u₀.(x_ic, y_ic), 1, :)
    v_ic = reshape(v₀.(x_ic, y_ic), 1, :)

    y_bc = rand(rng, T, bc_n)
    t_bc = rand(rng, T, bc_n)
    x0 = vcat(fill(xspan[1], bc_n)', y_bc', t_bc')
    x1 = vcat(fill(xspan[2], bc_n)', y_bc', t_bc')
    x_bc = rand(rng, T, bc_n)
    y0 = vcat(x_bc', fill(yspan[1], bc_n)', t_bc')
    y1 = vcat(x_bc', fill(yspan[2], bc_n)', t_bc')

    mins = T.([xspan[1], yspan[1], tspan[1]])
    maxs = T.([xspan[2], yspan[2], tspan[2]])
    normalizer = Normalizer(mins, maxs)
    denormalizer = inv(normalizer)

    norm = x -> normalizer(x) |> device_func

    xyt_pde = norm(xyt_pde)
    xyt_ic = norm(xyt_ic)
    x0 = norm(x0); x1 = norm(x1); y0 = norm(y0); y1 = norm(y1)
    u_ic = device_func(u_ic)
    v_ic = device_func(v_ic)

    model = PINNBrusselator()
    ps, st = Lux.setup(rng, model) |> device_func
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(T(0.001)))

    pde_loader = DataLoader(xyt_pde; batchsize=256, shuffle=true)
    ic_loader = DataLoader((xyt_ic, u_ic, v_ic); batchsize=256, shuffle=true)
    bc_loader = DataLoader((x0, x1, y0, y1); batchsize=128, shuffle=true)

    loss_trackers = ntuple(_ -> OnlineStats.CircBuff(T, 32), 4)
    max_iters = 50000
    lr = i -> i < 10000 ? T(0.001) : (i < 30000 ? T(0.0001) : T(1e-5))

    for (i, (xyt, ic, bc)) in enumerate(zip(Iterators.cycle(pde_loader), Iterators.cycle(ic_loader), Iterators.cycle(bc_loader)))
        Optimisers.adjust!(train_state.optimizer_state, lr(i))
        data = (xyt, ic, bc, denormalizer, α)

        loss, st_new, stats = Lux.Training.single_train_step!(AutoEnzyme(), loss_fn, data, train_state; return_gradients=Val(false))
        train_state = Lux.Training.TrainState(train_state.model, train_state.parameters, st_new, train_state.optimizer_state)

        fit!.(loss_trackers, (T(loss), T(stats.loss_pde), T(stats.loss_ic), T(stats.loss_bc)))

        if i % 1000 == 1 || i == max_iters
            m = mean ∘ OnlineStats.value
            @printf "Iter: %5d  Loss: %.6e  PDE: %.2e  IC: %.2e  BC: %.2e\n" i loss m(loss_trackers[2]) m(loss_trackers[3]) m(loss_trackers[4])
        end
        i ≥ max_iters && break
    end

    return train_state, normalizer, denormalizer
end

train_state, norm, denorm = train_brusselator!()

function visualize_brusselator(train_state, normalizer, denormalizer)
    xs = range(0f0, 1f0; length=50)
    ys = range(0f0, 1f0; length=50)
    ts = range(0f0, 11.5f0; length=40)

    grid = stack([[x, y, t] for t in ts, y in ys, x in xs])
    grid = reshape(permutedims(grid), 3, :)

    norm_grid = normalizer(grid) |> device_func

    u_net = StatefulLuxLayer{true}(train_state.model.u, cpu_device()(train_state.parameters.u), cpu_device()(train_state.states.u))
    v_net = StatefulLuxLayer{true}(train_state.model.v, cpu_device()(train_state.parameters.v), cpu_device()(train_state.states.v))

    u_pred, _ = u_net(norm_grid)
    v_pred, _ = v_net(norm_grid)

    u_pred = reshape(Array(u_pred), length(xs), length(ys), length(ts))
    v_pred = reshape(Array(v_pred), length(xs), length(ys), length(ts))

    fig_u = Figure(size=(800, 600))
    ax_u = Axis(fig_u[1, 1], xlabel="x", ylabel="y", title="U")
    umin, umax = extrema(u_pred)
    plt_u = heatmap!(ax_u, xs, ys, u_pred[:, :, 1]; colorrange=(umin, umax))
    Colorbar(fig_u[1, 2], plt_u, label="U")

    CairoMakie.record(fig_u, "brusselator_U.gif", 1:length(ts); framerate=10) do i
        plt_u[3] = u_pred[:, :, i]
        ax_u.title = "U Concentration | t = $(round(ts[i], digits=2))"
    end

    fig_v = Figure(size=(800, 600))
    ax_v = Axis(fig_v[1, 1], xlabel="x", ylabel="y", title="V")
    vmin, vmax = extrema(v_pred)
    plt_v = heatmap!(ax_v, xs, ys, v_pred[:, :, 1]; colorrange=(vmin, vmax))
    Colorbar(fig_v[1, 2], plt_v, label="V")

    CairoMakie.record(fig_v, "brusselator_V.gif", 1:length(ts); framerate=10) do i
        plt_v[3] = v_pred[:, :, i]
        ax_v.title = "V Concentration | t = $(round(ts[i], digits=2))"
    end

    println("Saved U to brusselator_U.gif and V to brusselator_V.gif")
end

visualize_brusselator(train_state, norm, denorm)