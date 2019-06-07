using ForwardDiff, Knet

sigm(x) =  1 ./ (1 .+ exp.(.-x))

function predict(P, x)
    w, b, v = P
    h = sigm(w * x .+ b)
    return v * h
end

function loss_trial(w, b, v, timepoints, f, p, phi)
    P = [w, b, v]
    dydx(P,x) = ForwardDiff.derivative(x -> phi(P, x), x)
    loss = sum(abs2, [dydx(P, x) - f(phi(P, x), p, x) for x in timepoints])
    return loss
end

function lossgradient(P, timepoints, f, p, phi)
    w, b, v = P
    loss∇w = ForwardDiff.gradient(w -> loss_trial(w, b, v, timepoints, f,p,phi), w)
    loss∇b = ForwardDiff.gradient(b -> loss_trial(w, b, v, timepoints, f,p,phi), b)
    loss∇v = ForwardDiff.gradient(v -> loss_trial(w, b, v, timepoints, f,p,phi), v)
    return (loss∇w, loss∇b, loss∇v)
end

function train(P, prms, timepoints, f,p,phi)
    g = lossgradient(P,timepoints,f,p,phi)
    update!(P, g, prms)
    return P
end

function test(P,timepoints,f,p,phi)
    sumloss = numloss = 0
    w, b, v = P
    for t in timepoints
        sumloss += loss_trial(w,b,v,t,f,p,phi)
        numloss += 1
    end
    return sumloss/numloss
end

function init_params(ftype,hl_width)
    P = Array{Any}(undef, 3)
    P[1] = randn(hl_width,1)
    P[2] = zeros(hl_width,1)
    P[3] = randn(1,hl_width)
    return P
end

function generate_data(low, high, dt; atype=KnetArray{Float32})
    num_points = floor(Int, 1/dt)
    x = range(low,stop=high,length = num_points+1) |> collect
    return x
end
