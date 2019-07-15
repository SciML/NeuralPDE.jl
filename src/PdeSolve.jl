function pde_solve(
    prob,
    grid,
    neuralNetworkParams;
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)


    x0 = grid[1] #initial points
    t0 = grid[2] #initial time
    tn = grid[3] #terminal time
    dt = grid[4] #time step
    d  = grid[5] # number of dimensions
    m =  grid[6] # number of trajectories (batch size)


    g(x) = prob[1](x)
    f(t,x,Y,Z)  = prob[2](t,x,Y,Z)
    μ(t,x) = prob[3](t,x)
    σ(t,x) = prob[4](t,x)


    data = Iterators.repeated((), maxiters)
    ts = t0:dt:tn

    #hidden layer
    hide_layer_size = neuralNetworkParams[1]
    opt = neuralNetworkParams[2]

    U0(hide_layer_size, d) = neuralNetworkParams[3](hide_layer_size, d)
    gradU(hide_layer_size, d) = neuralNetworkParams[4](hide_layer_size, d)

    chains = [gradU(hide_layer_size, d) for i=1:length(ts)]
    chainU = U0(hide_layer_size, d)
    ps = Flux.params(chainU, chains...)

    # brownian motion
    dw(dt) = sqrt(dt) * randn()
    # the Euler-Maruyama scheme
    x_sde(x_dim,t,dwa) = [x_dim[i] + μ(t,x_dim[i])*dt + σ(t,x_dim[i])*dwa[i] for i = 1: d]

    get_x_sde(x_cur,l,dwA) = [x_sde(x_cur[i], ts[l] , dwA[i]) for i = 1: m]
    reduceN(x_dim, l, dwA) = sum([gradU*dwA[i] for (i, gradU) in enumerate(chains[l](x_dim))])
    getN(x_cur, l, dwA) = [reduceN(x_cur[i], l, dwA[i]) for i = 1: m]

    x_0 = [x0 for i = 1: m]

    function sol()
        x_cur = x_0
        U = [chainU(x)[1] for x in x_0]
        global x_prev
        for l = 1 : length(ts)
            x_prev = x_cur
            dwA = [[dw(dt) for _=1:d] for _=1:m]
            fa = [f(ts[l], x_cur[i], U[i], chains[l](x_cur[i])) for i= 1 : m]
            U = U - fa*dt + getN(x_cur, l, dwA)
            x_cur = get_x_sde(x_cur,l,dwA)
        end
        (U, x_prev)
    end

    function loss()
        U0, x_cur = sol()
        return sum(abs2, g.(x_cur) .- U0) / m
    end


    cb = function ()
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = cb)

    ans = chainU(x0)[1]
    ans
end #pde_solve
