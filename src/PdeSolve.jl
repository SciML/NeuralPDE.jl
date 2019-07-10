
#  one-dimensional deep bsde pde solver
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


    x0 = grid[1]
    t0 = grid[2]
    tn = grid[3]
    dt = grid[4]
    d  = grid[5] # number of dimensions
    m =  grid[6] # number of trajectories (batch size)

    g(x) = prob[1](x)
    f(x,t) = prob[2](x,t)
    μ(x,t) = prob[3](x,t)
    σ(x,t) = prob[4](x,t)


    data = Iterators.repeated((), maxiters)
    ts = t0:dt:tn

    #hidden layer
    hide_layer_size = neuralNetworkParams[1]
    opt = neuralNetworkParams[2]
    getNeuranNetwork(hide_layer_size, d) = neuralNetworkParams[3](hide_layer_size, d)


    chains = [getNeuranNetwork(hide_layer_size, d) for i=1:length(ts)]
    chainU = getNeuranNetwork(hide_layer_size, d)
    ps = Flux.params(chainU, chains...)


    dw(dt) = sqrt(dt) * randn()
    x_sde(x_cur,t,dwa) = [x + μ(x, t)*dt + σ(x,t)*dwa for x in x_cur]
    get_x_sde(x_cur,l,dwA) =[x_sde(x_cur[i], ts[l] , dwA[i]) for i=1:length(x_cur)]
    reduceN(x_cur, l, dwA) = [chains[l](x_cur[i])[1]*dwA[i] for i=1:length(x_cur)]
    x_0 = [x0 for i = 1: m]

    function sol()
        x_cur = x_0
        U = [chainU(x)[1] for x in x_0]
        global x_cur
        for l = 1:length(ts)

            dwA = [dw(dt) for i= 1:length(x_cur)]
            fa = [f(x,ts[l]) for x in x_cur]
            U = U - fa*dt + reduceN(x_cur, l, dwA)
            x_cur = get_x_sde(x_cur,l,dwA)
        end
        (U, x_cur)
    end

    function loss()
        U0, x_cur = sol()
        return sum(abs2, g.(x_cur) .- U0)
    end


    cb = function ()
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = cb)

    ans = chainU(x0)[1]
    ans
end#solver
