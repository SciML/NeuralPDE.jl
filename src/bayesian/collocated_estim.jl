# suggested extra loss function
function L2loss2(Tar::LogTargetDensity, θ)
    f = Tar.prob.f

    # parameter estimation chosen or not
    if Tar.extraparams > 0
        dataset, deri_sol = Tar.dataset
        # deri_sol = deri_sol'
        autodiff = Tar.autodiff

        # # Timepoints to enforce Physics
        # dataset = Array(reduce(hcat, dataset)')
        # t = dataset[end, :]
        # û = dataset[1:(end - 1), :]

        # ode_params = Tar.extraparams == 1 ?
        #              θ[((length(θ) - Tar.extraparams) + 1):length(θ)][1] :
        #              θ[((length(θ) - Tar.extraparams) + 1):length(θ)]

        # if length(û[:, 1]) == 1
        #     physsol = [f(û[:, i][1],
        #         ode_params,
        #         t[i])
        #                for i in 1:length(û[1, :])]
        # else
        #     physsol = [f(û[:, i],
        #         ode_params,
        #         t[i])
        #                for i in 1:length(û[1, :])]
        # end
        # #form of NN output matrix output dim x n
        # deri_physsol = reduce(hcat, physsol)

        # > for perfect deriv(basically gradient matching in case of an ODEFunction)
        # in case of PDE or general ODE we would want to reduce residue of f(du,u,p,t)
        # if length(û[:, 1]) == 1
        #     deri_sol = [f(û[:, i][1],
        #         Tar.prob.p,
        #         t[i])
        #                 for i in 1:length(û[1, :])]
        # else
        #     deri_sol = [f(û[:, i],
        #         Tar.prob.p,
        #         t[i])
        #                 for i in 1:length(û[1, :])]
        # end
        # deri_sol = reduce(hcat, deri_sol) 
        # deri_sol = reduce(hcat, derivatives)

        # Timepoints to enforce Physics 
        t = dataset[end]
        u1 = dataset[2]
        û = dataset[1]
        # Tar(t, θ[1:(length(θ) - Tar.extraparams)])'
        #  

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
                       for i in 1:length(û[:, 1])]
        end
        #form of NN output matrix output dim x n 
        deri_physsol = reduce(hcat, physsol)

        # if length(Tar.prob.u0) == 1
        #     nnsol = [f(û[i],
        #         Tar.prob.p,
        #         t[i])
        #              for i in 1:length(û[:, 1])]
        # else
        #     nnsol = [f([û[i], u1[i]],
        #         Tar.prob.p,
        #         t[i])
        #              for i in 1:length(û[:, 1])]
        # end
        # form of NN output matrix output dim x n
        # nnsol = reduce(hcat, nnsol)

        # > Instead of dataset gradients trying NN derivatives with dataset collocation 
        # # convert to matrix as nnsol  

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

# PDE(DU,U,P,T)=0

# Derivated via Central Diff
# function calculate_derivatives2(dataset)
#     x̂, time = dataset
#     num_points = length(x̂)
#     # Initialize an array to store the derivative values.
#     derivatives = similar(x̂)

#     for i in 2:(num_points - 1)
#         # Calculate the first-order derivative using central differences.
#         Δt_forward = time[i + 1] - time[i]
#         Δt_backward = time[i] - time[i - 1]

#         derivative = (x̂[i + 1] - x̂[i - 1]) / (Δt_forward + Δt_backward)

#         derivatives[i] = derivative
#     end

#     # Derivatives at the endpoints can be calculated using forward or backward differences.
#     derivatives[1] = (x̂[2] - x̂[1]) / (time[2] - time[1])
#     derivatives[end] = (x̂[end] - x̂[end - 1]) / (time[end] - time[end - 1])
#     return derivatives
# end

function calderivatives(prob, dataset)
    chainflux = Flux.Chain(Flux.Dense(1, 8, tanh), Flux.Dense(8, 8, tanh),
        Flux.Dense(8, 2)) |> Flux.f64
    # chainflux = Flux.Chain(Flux.Dense(1, 7, tanh), Flux.Dense(7, 1)) |> Flux.f64
    function loss(x, y)
        # sum(Flux.mse.(prob.u0[1] .+ (prob.tspan[2] .- x)' .* chainflux(x)[1, :], y[1]) +
        #     Flux.mse.(prob.u0[2] .+ (prob.tspan[2] .- x)' .* chainflux(x)[2, :], y[2]))
        # sum(Flux.mse.(prob.u0[1] .+ (prob.tspan[2] .- x)' .* chainflux(x)[1, :], y[1]))
        sum(Flux.mse.(chainflux(x), y))
    end
    optimizer = Flux.Optimise.ADAM(0.01)
    epochs = 3000
    for epoch in 1:epochs
        Flux.train!(loss,
            Flux.params(chainflux),
            [(dataset[end]', dataset[1:(end - 1)])],
            optimizer)
    end

    # A1 = (prob.u0' .+
    #   (prob.tspan[2] .- (dataset[end]' .+ sqrt(eps(eltype(Float64)))))' .*
    #   chainflux(dataset[end]' .+ sqrt(eps(eltype(Float64))))')

    # A2 = (prob.u0' .+
    #       (prob.tspan[2] .- (dataset[end]'))' .*
    #       chainflux(dataset[end]')')

    A1 = chainflux(dataset[end]' .+ sqrt(eps(eltype(dataset[end][1]))))
    A2 = chainflux(dataset[end]')

    gradients = (A2 .- A1) ./ sqrt(eps(eltype(dataset[end][1])))

    return gradients
end

function calculate_derivatives(dataset)

    # u = dataset[1]
    # u1 = dataset[2]
    # t = dataset[end]
    # # control points
    # n = Int(floor(length(t) / 10))
    # # spline for datasetvalues(solution) 
    # # interp = BSplineApprox(u, t, 4, 10, :Uniform, :Uniform)
    # interp = CubicSpline(u, t)
    # interp1 = CubicSpline(u1, t)
    # # derrivatives interpolation
    # dx = t[2] - t[1]
    # time = collect(t[1]:dx:t[end])
    # smoothu = [interp(i) for i in time]
    # smoothu1 = [interp1(i) for i in time]
    # # derivative of the spline (must match function derivative) 
    # û = tvdiff(smoothu, 20, 0.5, dx = dx, ε = 1)
    # û1 = tvdiff(smoothu1, 20, 0.5, dx = dx, ε = 1)
    # # tvdiff(smoothu, 100, 0.035, dx = dx, ε = 1)
    # # FDM
    # # û1 = diff(u) / dx
    # # dataset[1] and smoothu are almost equal(rounding errors)
    # return [û, û1] 

end