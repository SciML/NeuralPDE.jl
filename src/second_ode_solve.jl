struct NNODE2{C,O} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
end
NNODE2(chain;opt=Flux.ADAM(0.1)) = NNODE2(chain,opt)

function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::NNODE2,
    args...;
    dt,
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100)

    u0, du0 = prob.u0
    tspan = prob.tspan
    f = prob.f.f1
    p = prob.p
    t0 = tspan[1]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    ps     = Flux.params(chain)
    data   = Iterators.repeated((), maxiters)

    #train points generation
    cnt = 0.0f0
    ts  = tspan[1]:dt:tspan[2]
    
    #Initial Value Problem
    phi(t) = (u0 .+ du0 .* (t .- tspan[1]) .+ (t .- tspan[1]) .^ 2 .* chain([t]))[1]


    dfdx = t -> (phi(t+sqrt(eps(typeof(dt)))) - phi(t)) / sqrt(eps(typeof(dt)))
    epsilon(t) = cbrt(eps(typeof(dt)))
    #second order central
    d2fdx2(t) = (phi(t+epsilon(t)) - 2phi(t) + phi(t-epsilon(t)))/epsilon(t)^2
    loss = () -> sum(abs2,sum(abs2,d2fdx2(t) .- f(dfdx(t), phi(t),p,t)[1]) for t in ts)

    cb = function ()
        l = loss()
        verbose && println("Current loss is: $l")
        l < abstol && Flux.stop()
    end
    Flux.train!(loss, ps, data, opt; cb = cb)

    #solutions at timepoints
    u = [phi(t) for t in ts]

    sol = DiffEqBase.build_solution(prob,alg,ts,u,calculate_error = false)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    sol
end #solve
