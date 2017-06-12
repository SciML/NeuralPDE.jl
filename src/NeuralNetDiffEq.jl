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
#z(P,i,x) = P[1][i]*x + P[2][i]
sig_der(x) = sigm(x)*(1-sigm(x))


function solve(
    prob::AbstractODEProblem,
    alg::NeuralNetDiffEqAlgorithm;
    dt = nothing,
    timeseries_errors = true,
    iterations = 50)

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


    #hidden layer
    hl_width = alg.hl_width

    #The phi trial solution
    phi(P,t) = u0 + (t-t0)*predict(P,t)


    #train points generation
    dtrn = generate_data(tspan[1],tspan[2],dt,atype=tType)

    #iterations
    _maxiters = iterations

    #initialization of weights and bias
    w = init_params(uElType,hl_width)

    #initialization of optimization parameters (Adam by default for now)
    lr_ = 0.06
    beta1_ = 0.9
    beta2_ = 0.95
    eps_ = 1e-6
    prms = Any[]

    for i=1:length(w)
    prm = Adam(lr=lr_, beta1=beta1_, beta2=beta2_, eps=eps_)
    #prm = Sgd(;lr=lr_)
    push!(prms, prm)
    end

    @time for iters=1:_maxiters
            train(w, prms, dtrn, f, phi, hl_width; maxiters=1)
            loss = loss_trial(w,dtrn,f,phi,hl_width)
            if mod(iters,100) == 0
                println((:iteration,iters,:loss,loss))
            end
            #gradcheck(loss_trial, w, dtrn, f, phi, hl_width...; gcheck=10, verbose=true)
            #check_Phi_gradient(w,dtrn,hl_width)
            if loss < 10^(-15.0)
                print(:loss,loss)
                break
            end
        end


    #solutions at timepoints
    u = [phi(w,x) for x in dtrn]


    build_solution(prob,alg,dtrn,u,
               timeseries_errors = timeseries_errors,
               retcode = :Success)

end #solve

include("training_utils.jl")
end # module
