using Flux, DiffEqFlux, StochasticDiffEq, LinearAlgebra

d = 3
hls = 10 + d
datasize = 5
tspan = (0.0f0,1.0f0)
t = range(tspan[1],tspan[2],length=datasize)
x0 = Float32[1.,1.,1.]
f(X, u, _σᵀ∇u, p, t) = Float32(0.05) * (u .- sum(X.*_σᵀ∇u))
μ(X,p,t) = 1.0
σ(X,p,t) = 1.0
g(X) = sum(X.^2)

u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))

σᵀ∇u = Flux.Chain(Dense(d,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))


function F(h,p,t)
    u =  h[end]
    X =  h[1:end-1]
    _σᵀ∇u = σᵀ∇u(X) #TODO time dependence
    _f = -f(X, u, _σᵀ∇u, p, t)
    Flux.Tracker.collect([Float32[μ(x,p,t) for x in X]; [_f]])
end

function G(h,p,t)
    X = h[1:end-1]
    _σᵀ∇u = σᵀ∇u(h[1:end-1]) #TODO time dependence
    Flux.Tracker.collect([Float32[σ(x,p,t) for x in x0]; [sum(_σᵀ∇u)]]) ## TODO _σᵀ∇u*dW nondioganal
end

# prob = SDEProblem(trueODEfunc,true_noise_func,init_cond,tspan) #,noise_rate_prototype=zeros(2,d)
# sol = solve(prob,SRIW1())
# println(sol)

function neural_sde(init_cond, F, G, tspan, args...; kwargs...)
    prob = SDEProblem(F,G, init_cond, tspan,nothing) #noise_rate_prototype=zeros(2,d)
    solve(prob, args...; kwargs...) |> Tracker.collect
end

n_sde = init_cond->neural_sde(init_cond,F,G,tspan,SOSRI(),saveat=t,reltol=1e-1,abstol=1e-1)

function predict_n_sde()
    _u0 = u0(x0)
    init_cond = [x0;_u0]
    n_sde(init_cond)
end

# init_cond = [Flux.data(_u0),Flux.data(_u0),Flux.data(_u0),Flux.data(_u0)]
# prob = ODEProblem(dudt_,x,tspan,p)
# prob = SDEProblem(trueODEfunc,true_noise_func, init_cond, tspan,nothing)
# diffeq_rd(p,prob,Tsit5())
# _init_cond = vcat(param(Float32[1., 1., 1.]),_u0)
# function predict_n_sde()
#     _u0 = u0(x0)
#     _init_cond = vcat(x0,_u0)
#     Flux.Tracker.collect(diffeq_rd(1,prob,SOSRI(),u0=_init_cond,saveat=t,reltol=1e-1,abstol=1e-1))
# end

# ans = predict_n_sde()
# println(ans)
# ans = predict_n_sde()[:,end]
# (X, u) = (ans[1:end-1],ans[end])
# println(ans)
# println((X, u))

function loss_n_sde()
    predict_ans = predict_n_sde()[:,end]
    (X, u) = (Flux.data(predict_ans[1:end-1]), predict_ans[end])
    sum(abs2,g(X) - u)
end

ps = Flux.params(u0, σᵀ∇u)
opt = ADAM(0.005)
data = Iterators.repeated((), 50)
verbose = true
abstol=1e-8

cb = function ()
    l = loss_n_sde()
    verbose && println("Current loss is: $l")
    l < abstol && Flux.stop()
end

cb()

Flux.train!(loss_n_sde , ps, data, opt, cb = cb)
