struct NNDAE{C, W, O, P, K} <: NeuralPDEAlgorithm
    chain::C
    W::W
    opt::O
    init_params::P
    autodiff::Bool
    kwargs::K
end


function NNDAE(chain, W, opt = .., init_params = nothing; autodiff = false,
        kwargs...)
   ...
    NNDAE(chain...)
end

function DiffEqBase.solve(prob::DiffEqBase.AbstractDAEProblem,
        alg::NeuralPDEAlgorithm,
        args...;
        dt,
        timeseries_errors = true,
        save_everystep = true,
        adaptive = false,
        abstol = 1.0f-6,
        verbose = false,
        maxiters = 100)
...
end #solve
