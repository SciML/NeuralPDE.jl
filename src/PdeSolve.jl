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

    X0 = grid[1] #initial points
    t0 = grid[2] #initial time
    tn = grid[3] #terminal time
    dt = grid[4] #time step
    d  = grid[5] # number of dimensions
    m =  grid[6] # number of trajectories (batch size)

    g = prob[1]
    f = prob[2]
    μ = prob[3]
    σ = prob[4]

    data = Iterators.repeated((), maxiters)
    ts = t0:dt:tn

    #hidden layer
    hide_layer_size = neuralNetworkParams[1]
    opt = neuralNetworkParams[2]

    u0 = neuralNetworkParams[3](hide_layer_size, d)
    σᵀ∇u = [neuralNetworkParams[4](hide_layer_size, d) for i=1:length(ts)]
    ps = Flux.params(u0, ∇u...)

    function sol()
        map(1:m) do
            u = u0(X0)
            X = X0
            for i in 1:length(ts)-1
                t = ts[i]
                _σᵀ∇u = σᵀ∇u[i](X)
                dW = sqrt(dt)*randn(d)
                u = u - f(t, X, u, _σᵀ∇u)*dt + _σᵀ∇u*dW
                X  = X + μ(t,X)*dt + σ(t,X)*dW
            end
            X,u
        end
    end

    function loss()
        mean(sum(abs2,g(X) - u) for (X,u) in sol())
    end


    cb = function ()
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end

    Flux.train!(loss, ps, data, opt; cb = cb)

    u0(X0)
end #pde_solve
