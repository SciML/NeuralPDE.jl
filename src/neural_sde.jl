using Flux, DiffEqFlux, StochasticDiffEq, LinearAlgebra, Statistics

#three-dimensional heat equation
d = 3
hls = 10 + d
tspan = (0.0f0,2.0f0)
dt = 0.2
t = tspan[1]:dt:tspan[2]
x0 = Float32[11.,11.,11.]
# f(X, u, _σᵀ∇u, p, t) = Float32(0.05) * (u .- sum(X.*_σᵀ∇u))
g(X) = sum(X.^2)   # terminal condition
f(X, u, _σᵀ∇u, p, t) = Float32(0.0)
μ(X,p,t) = zero(X) #Vector d*1
σ(X,p,t) = Diagonal(ones(Float32,d)) #Matrix d*d

u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))

σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))

function F(h, p, t)
    u =  h[end]
    X =  h[1:end-1]
    _σᵀ∇u = σᵀ∇u([X;t])
    _f = -f(X, u, _σᵀ∇u, p, t)
    [μ(X,p,t); _f]
end

function G(h, p, t)
    X = h[1:end-1]
    _σᵀ∇u = σᵀ∇u([X;t])'
    [σ(X,p,t);_σᵀ∇u]
end

trajectories = 70
function neural_sde(init_cond, F, G, tspan, args...; kwargs...)
    noise = zeros(Float32,d+1,d)
    prob = SDEProblem(F, G, init_cond, tspan, noise_rate_prototype=noise)
    map(1:trajectories) do j #TODO add Ensemble Simulation
        predict_ans = solve(prob,  args...; kwargs...)[end]
        (X,u) = (predict_ans[1:end-1], predict_ans[end])
    end
end

n_sde = init_cond->neural_sde(init_cond,F,G,tspan, EM(), dt=dt, saveat=t,reltol=1e-1,abstol=1e-1)

function predict_n_sde()
    _u0 = u0(x0)
    init_cond = [x0;_u0]
    n_sde(init_cond)
end


function loss_n_sde()
    mean(sum(abs2, g(X) - u) for (X,u) in predict_n_sde())
end

ps = Flux.params(u0,σᵀ∇u)
opt = ADAM(0.005)
data = Iterators.repeated((), 250)
verbose = true
abstol=1e-8

cb = function ()
    l = loss_n_sde()
    verbose && println("Current loss is: $l")
    l < abstol && Flux.stop()
end

cb()

Flux.train!(loss_n_sde, ps, data, opt, cb = cb)

u_analytical(x,t) = sum(x.^2) .+ d*t
analytical_ans = u_analytical(x0, tspan[end])
ans =u0(x0)[1].data
println(ans, " ", analytical_ans)
println(abs(ans - analytical_ans))
