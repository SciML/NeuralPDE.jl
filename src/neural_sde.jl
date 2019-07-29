using Flux, DiffEqFlux, StochasticDiffEq,LinearAlgebra
d = 3
hls = 10 + d
# _u0 = Flux.data(u0(x0)[1])


datasize = 5
tspan = (0.0f0,1.0f0)
t = range(tspan[1],tspan[2],length=datasize)

f(X, u, _σᵀ∇u, p, t) = 0.5 * (u .- sum(X.*_σᵀ∇u))
μ(X,p,t) = 1.0
σ(X,p,t) = 1.0

u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))

σᵀ∇u = Flux.Chain(Dense(d,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
g(X) = sum(X.^2)

function trueODEfunc(dh,h,p,t)
    u =  h[end]
    X =  h[1:end-1]
    _σᵀ∇u = σᵀ∇u(X)
    dh[end] = Flux.data(f(X, u, _σᵀ∇u, p, t))
    dh[1:end-1] .= μ(X,p,t)
    dh
end

function true_noise_func(dh,h,p,t)
    X = h[1:end-1]
    _σᵀ∇u = Flux.data(σᵀ∇u(X))
    dh[end] = sum(_σᵀ∇u) # TODO  make _σᵀ∇u*dW nondioganal
    dh[1:end-1] .= σ(X,p,t)
    dh
end

# prob = SDEProblem(trueODEfunc,true_noise_func,init_cond,tspan) #,noise_rate_prototype=zeros(2,d)
# sol = solve(prob,SRIW1())
# println(sol)

function neural_sde(init_cond, trueODEfunc, true_noise_func, tspan, args...; kwargs...)
    prob = SDEProblem(trueODEfunc,true_noise_func, init_cond, tspan,nothing) #,noise_rate_prototype=zeros(2,d)
    solve(prob, args...; kwargs...) |> Tracker.collect
end

n_sde = init_cond->neural_sde(init_cond,trueODEfunc,true_noise_func,tspan,SOSRI(),saveat=t,reltol=1e-1,abstol=1e-1)

x0 = fill(1.0 , d)
init_cond = fill(1.0 , d+1)

function predict_n_sde()
    _u0 = Flux.data(u0(x0)[1])
    init_cond[d+1] =_u0
    ans = n_sde(init_cond)
end

# ans = predict_n_sde()[:,end]
# (X, u) = (ans[1:end-1],ans[end])
# println(ans)
# println((X, u))

function loss_n_sde()
    ans = predict_n_sde()[:,end]
    (X, u) = (ans[1:end-1],ans[end])
    sum(abs2,g(X) - u)
end


ps = Flux.params(u0, σᵀ∇u)

opt = ADAM(0.025)
data = Iterators.repeated((), 100)

verbose = true
abstol=1e-8

cb = function ()
    l = loss_n_sde()
    verbose && println("Current loss is: $l")
    l < abstol && Flux.stop()
end

cb()

Flux.train!(loss_n_sde , ps, data, opt, cb = cb)
