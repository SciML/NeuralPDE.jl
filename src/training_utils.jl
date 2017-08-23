#Utility functions for solver

function tr_sin(x::Any)
    y = Array{Any}(length(x))
    for i=1:length(x)
        if x[i]<(-pi/2)
            y[i] = 0
        elseif -pi/2 <= x[i] <= pi/2
            y[i] = sin(x[i])
        else
            y[i] = 1
        end
    end
    return y
end



function predict(w,x)
    #println(typeof(w),typeof(x))
    for i=1:2:(length(w)-1)
        x = w[i]*x .+ w[i+1]
        if i<length(w)-1
            x = tanh(x) # max(0,x)
        end
    end
    return w[end]*x
end

function get_trial_sol_values(trial_solutions,NN,t)
    trial_sol_values = Array{Any}(length(trial_solutions))
    for i = 1:length(trial_solutions)
        trial_sol_values[i] = trial_solutions[i](NN,t)
    end
    trial_sol_values
end

function loss_trial(NN,timepoints,f,trial_solutions,hl_width,outdim)
    sum([sumabs2([gradient(x->trial_solutions[i](NN,x),t) .- f(t,[trial_func(NN,t) for trial_func in trial_solutions])[i]  for t in timepoints]) for i =1:outdim])
end

lossgradient = grad(loss_trial)

function train(NN, prms, timepoints, f, trial_solutions, hl_width,outdim; maxiters =1)
        for x in timepoints
                g = lossgradient(NN,timepoints,f,trial_solutions,hl_width,outdim)
                update!(NN, g, prms)
        end
    return NN
end


function init_weights_and_biases(ftype,hl_width,outdim;atype=KnetArray{Float32})
    num_weights_and_bias = 2*length(hl_width) + 1
    P = Array{Any}(num_weights_and_bias) #Constant layers and parameters for now
    hidden_layer = 1
    #println(size(hl_width))
    for i = 1:2:(num_weights_and_bias-1)
        #println(typeof(hl_width),hidden_layer,num_weights_and_bias,i)
        P[i] = randn(ftype,hl_width[hidden_layer],hidden_layer > 1 ? hl_width[hidden_layer-1] : 1)*(0.01^2)  #To reduce variance
        P[i+1] = zeros(ftype,hl_width[hidden_layer],1)
        hidden_layer = hidden_layer + 1

    end
    P[num_weights_and_bias] = randn(ftype,outdim,hl_width[end])*(0.01^2)  #To reduce variance
    return P
end


function generate_data(low, high, dt; atype=KnetArray{Float32})
    num_points = (high-low)/dt
    x = linspace(low,high,num_points)
    return x
end
