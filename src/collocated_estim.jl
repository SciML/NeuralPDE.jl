# suggested extra loss function for ODE solver case
function L2loss2(Tar::LogTargetDensity, θ)
    f = Tar.prob.f

    # parameter estimation chosen or not
    if Tar.extraparams > 0
        autodiff = Tar.autodiff
        # Timepoints to enforce Physics 
        t = Tar.dataset[end]
        u1 = Tar.dataset[2]
        û = Tar.dataset[1]

        nnsol = NNodederi(Tar, t, θ[1:(length(θ) - Tar.extraparams)], autodiff)

        ode_params = Tar.extraparams == 1 ?
                     θ[((length(θ) - Tar.extraparams) + 1):length(θ)][1] :
                     θ[((length(θ) - Tar.extraparams) + 1):length(θ)]

        if length(Tar.prob.u0) == 1
            physsol = [f(û[i],
                ode_params,
                t[i])
                       for i in 1:length(û[:, 1])]
        else
            physsol = [f([û[i], u1[i]],
                ode_params,
                t[i])
                       for i in 1:length(û)]
        end
        #form of NN output matrix output dim x n 
        deri_physsol = reduce(hcat, physsol)
   
        physlogprob = 0
        for i in 1:length(Tar.prob.u0)
            # can add phystd[i] for u[i] 
            physlogprob += logpdf(MvNormal(deri_physsol[i, :],
                    LinearAlgebra.Diagonal(map(abs2,
                        (Tar.l2std[i] * 4.0) .*
                        ones(length(nnsol[i, :]))))),
                nnsol[i, :])
        end
        return physlogprob
    else
        return 0
    end
end