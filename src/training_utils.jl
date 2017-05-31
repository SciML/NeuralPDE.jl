

function predict(P,x)
    w1, b1, w2 = P
    h = sigm(w1 * x .+ b1)
    return w2 * h
end

#when we have analytical solution available

# function loss_reg(w,x,ygold)
#     sumabs2(ygold - predict(w,x)) / size(ygold,2)
# end

function loss_trial(P,t,timepoints,f,phi,hl_width)
    w,b,v = P
    dN_dt(P,t) = sum([v[i]*w[i]*sig_der(w[i]*t .+ b[i]) for i = 1:hl_width])
    dPhi_dt(P,t) = predict(P,t)+t*dN_dt(P,t)
    sumabs2([dPhi_dt(P,t) - f(t,phi(P,t)) for t in timepoints][1])

end

lossgradient = grad(loss_trial)

function train(P, prms, timepoints, f, phi, hl_width; maxiters =100)
    #print(size(P),size(g),size(prms))



    for iter=1:maxiters
        #println("epoch no.",epoch)
        for x in timepoints
            g = lossgradient(P,x,timepoints,f,phi,hl_width)
          #print(size(P),size(g),size(prms))
          update!(P, g, prms)
          #println(P[1][1],P[2][1],P[3][1],P[4][1])
          #println(loss_trial(P,x))
        end

    end
    return P
end

function test(P,timepoints,f,phi,hl_width)
    sumloss = numloss = 0
    for t in timepoints
        sumloss += loss_trial(P,t,timepoints,f,phi,hl_width)
        numloss += 1
    end
    return sumloss/numloss
end


function init_params(ftype,hl_width;atype=KnetArray{Float32})
    #P = Vector{Vector{Float32}}(4)
    P = Array{Any}(3)
    P[1] = randn(ftype,hl_width,1)
    P[2] = zeros(ftype,hl_width,1)
    P[3] = randn(ftype,1,hl_width)
    #P[4] = zeros(Float32,1,1)
    #P = map(x -> convert(atype, x), P)

    return P
end




function generate_data(low, high, dt; atype=KnetArray{Float32})
    #data = Any[]
    num_points = 1/dt
    x = linspace(low,high,num_points+1)
    #x_ = convert(atype,x)
    # for i=1:num_points
    #     push!(data, (x_[i],y_[i]))
    # end
    return x
end
