__precompile__()

module NeuralNetDiffEq
#dependencies
using Knet, DiffEqBase, Compat
import DiffEqBase: solve

# Abstract Types
@compat abstract type NeuralNetDiffEqAlgorithm <: AbstractODEAlgorithm end

# DAE Algorithms
immutable nnode <: NeuralNetDiffEqAlgorithm
  hl_width::Int
end


#nnode(;hl_width=10) = nnode(hl_width)

export nnode
# package code goes here

#constant functions
z(P,i,x) = P[1][i]*x + P[2][i]
sig_der(x) = sigm(x)*(1-sigm(x))


function solve(
  prob::AbstractODEProblem,
  alg::NeuralNetDiffEqAlgorithm;
  dt = nothing)

  u0 = prob.u0
  tspan = prob.tspan
  global f = prob.f

  if dt == nothing
    dt = 1/100
  end


  #The phi trial solution
  phi(P,x) = u0 + x*(predict(P,x))


  #train and test points generation
  dtrn = generate_data(tspan[1],tspan[2],dt)
  #dtst =
  #println(dtrn)

  #initialization of weights and bias
  w = init_params()
  # println(w[1])
  # println(w[2])
  # println(w[3])

  #initialization of optimization parameters
  lr_ = 0.1
  beta1_ = 0.9
  beta2_ = 0.95
  eps_ = 1e-6
  prms = Any[]

  for i=1:length(w)
    prm = Adam(lr=lr_, beta1=beta1_, beta2=beta2_, eps=eps_)
    push!(prms, prm)
  end

  #flag to choose the loss
  regFlag = true

  #iters = 1000

  #reporting the accuracy
  #report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))
  #report(epoch)=println((:epoch,epoch,:trn,loss_trial(w,dtrn)))
  #report(0)
  #P_tuned = train(w,prms,dtrn,regFlag; epochs=100, iters=1000)
  epochs = 20
  @time for epoch=1:epochs

    train(w, prms, dtrn; epochs=20)
    #report(epoch)
    #(iters -= length(dtrn)) <= 0 && break
  end

  # for t in log; println(t); end
  return w

  build_solution(prob,alg,t,u,
               timeseries_errors = timeseries_errors,
               retcode = :Success)

end #solve

include("training_utils.jl")
#include("interface.jl")
end # module
