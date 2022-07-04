struct NNRODE{C, W, O, P, K} <: NeuralPDEAlgorithm
    chain::C
    W::W
    opt::O
    init_params::P
    autodiff::Bool
    kwargs::K
end
function NNRODE(chain, W, opt = Optim.BFGS(), init_params = nothing; autodiff = false,
                kwargs...)
    if init_params === nothing
        if chain isa Flux.Chain
            init_params, re = Flux.destructure(chain)
        else
            error("Only Flux is support here right now")
        end
    else
        init_params = init_params
    end
    NNRODE(chain, W, opt, init_params, autodiff, kwargs)
end

function DiffEqBase.solve(prob::DiffEqBase.AbstractRODEProblem,
                          alg::NeuralPDEAlgorithm,
                          args...;
                          dt,
                          timeseries_errors = true,
                          save_everystep = true,
                          adaptive = false,
                          abstol = 1.0f-6,
                          verbose = false,
                          maxiters = 100)
    DiffEqBase.isinplace(prob) && error("Only out-of-place methods are allowed!")

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t0 = tspan[1]

    #hidden layer
    chain = alg.chain
    opt = alg.opt
    autodiff = alg.autodiff
    Wg = alg.W
    #train points generation
    ts = tspan[1]:dt:tspan[2]
    init_params = alg.init_params

    if chain isa FastChain
        #The phi trial solution
        if u0 isa Number
            phi = (t, W, θ) -> u0 +
                               (t - tspan[1]) *
                               first(chain(adapt(DiffEqBase.parameterless_type(θ), [t, W]),
                                           θ))
        else
            phi = (t, W, θ) -> u0 +
                               (t - tspan[1]) *
                               chain(adapt(DiffEqBase.parameterless_type(θ), [t, W]), θ)
        end
    else
        _, re = Flux.destructure(chain)
        #The phi trial solution
        if u0 isa Number
            phi = (t, W, θ) -> u0 +
                               (t - t0) *
                               first(re(θ)(adapt(DiffEqBase.parameterless_type(θ), [t, W])))
        else
            phi = (t, W, θ) -> u0 +
                               (t - t0) *
                               re(θ)(adapt(DiffEqBase.parameterless_type(θ), [t, W]))
        end
    end

    if autodiff
        # dfdx = (t,W,θ) -> ForwardDiff.derivative(t->phi(t,θ),t)
    else
        dfdx = (t, W, θ) -> (phi(t + sqrt(eps(t)), W, θ) - phi(t, W, θ)) / sqrt(eps(t))
    end

    function inner_loss(t, W, θ)
        sum(abs, dfdx(t, W, θ) - f(phi(t, W, θ), p, t, W))
    end
    Wprob = NoiseProblem(Wg, tspan)
    Wsol = solve(Wprob; dt = dt)
    W = NoiseGrid(ts, Wsol.W)
    function loss(θ)
        sum(abs2, inner_loss(ts[i], W.W[i], θ) for i in 1:length(ts)) # sum(abs2,phi(tspan[1],θ) - u0)
    end

    callback = function (p, l)
        Wprob = NoiseProblem(Wg, tspan)
        Wsol = solve(Wprob; dt = dt)
        W = NoiseGrid(ts, Wsol.W)
        verbose && println("Current loss is: $l")
        l < abstol
    end
    #res = DiffEqFlux.sciml_train(loss, init_params, opt; cb = callback, maxiters = maxiters,
    #                             alg.kwargs...)

    #solutions at timepoints
    noiseproblem = NoiseProblem(Wg, tspan)
    W = solve(noiseproblem; dt = dt)
    if u0 isa Number
        u = [(phi(ts[i], W.W[i], res.minimizer)) for i in 1:length(ts)]
    else
        u = [(phi(ts[i], W.W[i], res.minimizer)) for i in 1:length(ts)]
    end

    sol = DiffEqBase.build_solution(prob, alg, ts, u, W = W, calculate_error = false)
    DiffEqBase.has_analytic(prob.f) &&
        DiffEqBase.calculate_solution_errors!(sol; timeseries_errors = true,
                                              dense_errors = false)
    sol
end #solve
