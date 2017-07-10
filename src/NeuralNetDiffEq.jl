__precompile__()

module NeuralNetDiffEq
#dependencies
using Knet, DiffEqBase, Compat, ForwardDiff
import DiffEqBase: solve

# Abstract Types
@compat abstract type NeuralNetDiffEqAlgorithm <: AbstractODEAlgorithm end

# DAE Algorithms
immutable nnode <: NeuralNetDiffEqAlgorithm
    hl_width::Int
end



nnode(;hl_width=10) = nnode(hl_width)



export nnode

#constant functions
sig_der(x) = sigm(x)*(1-sigm(x))


function solve(
    prob::AbstractODEProblem,
    alg::NeuralNetDiffEqAlgorithm;
    dt = nothing,
    timeseries_errors = true,
    iterations = 50,
    kwargs...)

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    t0 = tspan[1]

    if dt == nothing
        error("dt must be set.")
    end

    #types and dimensions
    uElType = eltype(u0)
    tType = typeof(tspan[1])
    outdim = length(u0)


    #hidden layer(s)
    hl_width = alg.hl_width

    #The trial solutions (one for each NN or ODE)
    trial_solutions = Array{Function}(outdim)
    for i = 1:outdim
        u(P,t) = u0[i] + (t .- t0).*predict(P,t)[1]
        trial_solutions[i] = u
    end

    #train points generation
    dtrn = generate_data(tspan[1],tspan[2],dt,atype=tType)

    #iterations
    _maxiters = iterations

    #initialization of weights and bias
    NNs = Array{Any}(outdim) #Array of Neural Nets each with w1, b1 and w2
    for i = 1:outdim
        NNs[i] = init_weights_and_biases(uElType,hl_width)
    end

    #initialization of optimization parameters (Adam by default for now)
    lr_ = 0.1
    beta1_ = 0.9
    beta2_ = 0.95
    eps_ = 1e-6
    prms = Any[]
    Params = Array{Any}(outdim)
    for i=1:length(NNs[1])
        prm = Adam(lr=lr_, beta1=beta1_, beta2=beta2_, eps=eps_)
        #prm = Sgd(;lr=lr_)
        push!(prms, prm)
    end

    for i=1:length(NNs)
        Params[i] = copy(prms)
    end


    @time for iters=1:_maxiters
            train(NNs, Params, dtrn, f, trial_solutions, hl_width; maxiters=1)

            loss = loss_trial(NNs,dtrn,f,trial_solutions,hl_width)
            if mod(iters,100) == 0
                println((:iteration,iters,:loss,loss))
            end
            if maximum(loss) < 10^(-15.0)
                print(:loss,loss)
                break
            end
        end


    #solutions at timepoints
    u = [get_trial_sol_values(trial_solutions,NNs,x) for x in dtrn]


    build_solution(prob,alg,dtrn,u,
               timeseries_errors = timeseries_errors,
               retcode = :Success)

end #solve

include("training_utils.jl")
end # module
