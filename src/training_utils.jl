#Utility functions for solver

function predict(P,x)
    w1, b1, w2 = P
    return w2*sigm(w1 * x .+ b1)
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


function init_weights_and_biases(ftype,hl_width;atype=KnetArray{Float32})
    P = Array{Any}(3) #Constant layers and parameters for now
    P[1] = randn(ftype,hl_width,1)*(0.01^2)  #To reduce variance
    P[2] = zeros(ftype,hl_width,1)
    P[3] = randn(ftype,1,hl_width)*(0.01^2)  #To reduce variance
    return P
end


function generate_data(low, high, dt; atype=KnetArray{Float32})
    num_points = (high-low)/dt
    x = linspace(low,high,num_points)
    return x
end
