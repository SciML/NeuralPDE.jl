#Utility functions for solver

function predict(w,x)
    #println(typeof(w),typeof(x))
    for i=1:2:(length(w)-1)
        x = w[i]*x .+ w[i+1]
        if i<length(w)-1
            x = sigm(x) # max(0,x)
        end
    end
    return w[length(w)]*x
end

function get_trial_sol_values(trial_solutions,NNs,t)
    trial_sol_values = Array{Any}(length(NNs))
    for i = 1:length(NNs)
        trial_sol_values[i] = trial_solutions[i](NNs[i],t)
    end
    trial_sol_values
end

function loss_trial(NNs,timepoints,f,trial_solutions,hl_width)
    sum([sumabs2([gradient(x->trial_solutions[i](NNs[i],x),t) .- f(t,[trial_func(NNs[i],t) for trial_func in trial_solutions])[i]  for t in timepoints]) for i =1:length(NNs)])
end

lossgradient = grad(loss_trial)

function train(NNs, prms, timepoints, f, trial_solutions, hl_width; maxiters =1)
        for x in timepoints
                g = lossgradient(NNs,timepoints,f,trial_solutions,hl_width)
                update!(NNs, g, prms)
        end
    return NNs
end


function init_weights_and_biases(ftype,hl_width,outdim;atype=KnetArray{Float32})
    weights_and_bias = 2*length(hl_width) + 1
    P = Array{Any}(weights_and_bias) #Constant layers and parameters for now
    hidden_layer = 1
    #println(size(hl_width))
    for i = 1:2:(weights_and_bias-1)
        #println(typeof(hl_width),hidden_layer,weights_and_bias,i)
        P[i] = randn(ftype,hl_width[hidden_layer],hidden_layer > 1 ? hl_width[hidden_layer-1] : 1)*(0.01^2)  #To reduce variance
        P[i+1] = zeros(ftype,hl_width[hidden_layer],1)
        hidden_layer = hidden_layer + 1

    end
    P[weights_and_bias] = randn(ftype,outdim,hl_width[hidden_layer-1])*(0.01^2)  #To reduce variance
    return P
end


function generate_data(low, high, dt; atype=KnetArray{Float32})
    num_points = (high-low)/dt
    x = linspace(low,high,num_points)
    return x
end
