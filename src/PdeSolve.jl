using Flux

#  one-dimensional deep bsde pde solver
function pde_solve(
    prob,
    grid;
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)


    # grid = (x0, xn, dx, t0, tn, dt, d)
    x0 = grid[1]
    xn = grid[2]
    dx = grid[3]
    t0 = grid[4]
    tn = grid[5]
    dt = grid[6]
    d = grid[7] #dimention

    # prob = (g, U0, f, μ, σ)
    g(x) = prob[1](x) #x.^2 .+ d*tn
    U0(x) = prob[2](x) #x.^2
    f(x,t) = prob[3](x,t) #0
    μ(x, t) = prob[4](x,t) #0
    σ(x,t) = prob[5](x,t)  #1

    #hidden layer
    hide_layer_size =d+10
    chain = Flux.Chain(Dense(d,hide_layer_size,relu),Dense(hide_layer_size,hide_layer_size,relu),
                Dense(hide_layer_size,hide_layer_size,relu),Dense(hide_layer_size,d))
    opt = Flux.ADAM(0.1, (0.9, 0.95))
    ps = Flux.params(chain)
    data = Iterators.repeated((), maxiters)


    ts = t0:dt:tn
    xd = x0:dx:xn
    dw(dt) = sqrt(dt) * randn()
    x_sde(x_prev,t) = x_prev .+ μ(x_prev, t)*dt .+ σ(x_prev,t)*dw(dt)
    N(x) = chain(x)
    reduceN(x_cur)  =  [N([x])[1] for x in x_cur]

    x_0 = [x for x in xd]
    U_0 = U0.(x_0)

    function Un()
        U = U_0
        x_prev = x_0
        for t in ts
            x_cur = x_sde(x_prev,t)
            # println(dw(dt))
            U = U .- f(x_prev,t)*dt .+ reduceN(x_cur)*(dt)
            x_prev = x_cur
        end
        U
    end

    loss = () -> sum(abs2, g(x_0) - Un())

    cb = function ()
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = cb)

    U = U0.(x_0)
    ans  = []
    for t in ts
        U = U .- f(x_0,t)*dt + reduceN(x_0)*dt
        push!(ans, U)
    end
    (ans, x_0, ts)
end#solver
