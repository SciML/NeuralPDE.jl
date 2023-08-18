# HIGH level API for BPINN ODE AND PDE SOLVER
struct BNNODE <: NeuralPDEAlgorithm
    dataset
    chain
end

struct BNNPDE <: NeuralPDEAlgorithm
    dataset
    chain
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem,
                            alg::BNNODE,args...;
                            )
    chain, samples, statistics= ahmc_bayesian_pinn_ode(prob, chain;
        dataset=[[]],
        init_params=nothing, draw_samples=1000,
        physdt=1 / 20.0, l2std=[0.05],
        phystd=[0.05], priorsNNw=(0.0, 2.0),
        param=[], nchains=1,
        autodiff=false,
        Kernel=HMC, Integrator=Leapfrog,
        Adaptor=StanHMCAdaptor, targetacceptancerate=0.8,
        Metric=DiagEuclideanMetric, jitter_rate=3.0,
        tempering_rate=3.0, max_depth=10, Δ_max=1000,
        n_leapfrog=10, δ=0.65, λ=0.3, progress=false,
        verbose=false)

        chain, samples, statistics= ahmc_bayesian_pinn_pde(pde_system, discretization;
    dataset=[[]],
    init_params=nothing, nchains=1,
    draw_samples=1000, l2std=[0.05],
    phystd=[0.05], priorsNNw=(0.0, 2.0),
    param=[],
    autodiff=false, physdt=1 / 20.0f0,
    Proposal=StaticTrajectory,
    Adaptor=StanHMCAdaptor, targetacceptancerate=0.8,
    Integrator=Leapfrog,
    Metric=DiagEuclideanMetric)

        if BNNODE.chain isa Lux.AbstractExplicitLayer
            θinit, st = Lux.setup(Random.default_rng(), chainlux1)
            θ = [vector_to_parameters(fhsamples2[i][1:(end - 1)], θinit) for i in 2000:2500]
            luxar = [chainlux1(t', θ[i], st)[1] for i in 1:500]
            luxmean = [mean(vcat(luxar...)[:, i]) for i in eachindex(t)]
            meanscurve2 = prob.u0 .+ (t .- prob.tspan[1]) .* luxmean
        else if BNNODE.chain isa Flux.Chain
            init1, re1 = destructure(chainflux1)
            out = re1.([fhsamples1[i][1:22] for i in 2000:2500])
            yu = collect(out[i](t') for i in eachindex(out))
            fluxmean = [mean(vcat(yu...)[:, i]) for i in eachindex(t)]
            meanscurve1 = prob.u0 .+ (t .- prob.tspan[1]) .* fluxmean
        else
            error("Only Lux.AbstractExplicitLayer and Flux.Chain neural networks are supported")
        end

        sol
end