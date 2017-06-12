

function predict(P,x)
    w1, b1, w2 = P
    h = sigm(w1 * x .+ b1)
    return w2 * h
end


function loss_trial(P,timepoints,f,phi,hl_width)
    w,b,v = P
    t0 = timepoints[1]
    sumabs2([gradient(x->phi(P,x),t) - f(t,phi(P,t)) for t in timepoints])[1]
end

lossgradient = grad(loss_trial)

function train(P, prms, timepoints, f, phi, hl_width; maxiters =100)
    for iter=1:maxiters
        for x in timepoints
            g = lossgradient(P,timepoints,f,phi,hl_width)
            update!(P, g, prms)
        end
    end
    return P
end


function init_params(ftype,hl_width;atype=KnetArray{Float32})
    #P = Vector{Vector{Float32}}(4)
    P = Array{Any}(3)
    P[1] = randn(ftype,hl_width,1)*(0.01^2)  #To reduce variance
    P[2] = zeros(ftype,hl_width,1)
    P[3] = randn(ftype,1,hl_width)*(0.01^2)  #To reduce variance
    #P[4] = zeros(Float32,1,1)
    #P = map(x -> convert(atype, x), P)

    return P
end


function generate_data(low, high, dt; atype=KnetArray{Float32})
    num_points = 1/dt
    x = linspace(low,high,num_points+1)
    return x
end
