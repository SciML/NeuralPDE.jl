

function predict(P,x)
    w1, b1, w2 = P
    h = sigm(w1 * x .+ b1)
    return w2 * h
end

function get_trial_sols(trial_funcs,NNs,t)
    trial_sols = Array{Any}(length(NNs))
    #println(length(NNs))
    for i = 1:length(NNs)
        #println(length(NNs))
        #println(i)
        T = trial_funcs[i](NNs[i],t)
        trial_sols[i] = T
    end
    #T1 = trial_sols[1](NNs[1],t)
    #T2 = trial_sols[2](P,t)
    trial_sols
end

function loss_trial(NNs,timepoints,f,trial_funcs,hl_width)
    sum([sumabs2([gradient(x->trial_funcs[i](NNs[i],x),t) .- f(t,[trial_func(NNs[i],t) for trial_func in trial_funcs])[i]  for t in timepoints]) for i =1:length(NNs)])
end

lossgradient = grad(loss_trial)

function train(NNs, prms, timepoints, f, trial_funcs, hl_width; maxiters =1)
        for x in timepoints
            #for i = 1:length(NNs)
                g = lossgradient(NNs,timepoints,f,trial_funcs,hl_width)
                update!(NNs, g, prms)
            #end
        end
    return NNs
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
    num_points = (high-low)/dt
    x = linspace(low,high,num_points)
    return x
end
