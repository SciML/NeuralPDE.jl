using NeuralPDE
using Test, Flux
using StochasticDiffEq
using LinearAlgebra

d = 1
m = Chain(Dense(3, 16, elu), Dense(16, 16, elu), Dense(16, 5, elu), Dense(5, 1))
ensemblealg = EnsembleThreads()
γ_mu_prototype = nothing
γ_sigma_prototype = zeros(d, d, 1)
γ_phi_prototype = nothing

sdealg = EM()
tspan = (0.00, 1.00)
trajectories = 10000
function phi(x, y_phi)
    x .^ 2
end
g(x, γ_sigma) = γ_sigma[:, :, 1]
f(x, γ_mu_1, γ_mu_2) = [0.00]
xspan = (0.00, 3.00)
y_domain = KolmogorovParamDomain((0.00, 2.00), (0.00, 0.00), (0.00, 0.00))
dt = 0.01
dx = 0.01
dy = 0.01
opt = Flux.ADAM(1e-2)
prob = ParamKolmogorovPDEProblem(f, g, phi, xspan, tspan, d, y_domain;
                                 Y_sigma_prototype = γ_sigma_prototype)
sol = solve(prob, NNParamKolmogorov(m, opt, sdealg, ensemblealg), verbose = true, dt = 0.01,
            abstol = 1e-10, dx = 0.01, trajectories = trajectories, maxiters = 400,
            use_gpu = false)

x_test = rand(xspan[1]:dx:xspan[2], d, 1, 1000)
t_test = rand(tspan[1]:dt:tspan[2], 1, 1000)
γ_sigma_test = rand(0.3:dy:0.3, d, d, 1, 1000)

function getTestData(x_test, t_test, γ_sigma_test)
    X_test = reshape([], 3, 0)
    for i in 1:length(t_test)
        K = vcat(t_test[i], x_test[:, :, i],
                 reshape(γ_sigma_test[:, :, :, i], d^2 * (size(γ_sigma_prototype))[3], 1))
        X_test = hcat(X_test, K)
    end
    return X_test
end

X_test = getTestData(x_test, t_test, γ_sigma_test)

function analytical(x, t, y)
    #return sum(x.^2) + t*tr(y*y) for multidimensional
    return x^2 + t * (y * y)
end
function y_analytical(X)
    y_test = reshape([], 1, 0)
    for i in 1:length(t_test)
        y_test = hcat(y_test,
                      analytical(X[:, i][2], X[:, i][1], γ_sigma_test[:, :, 1, i][1]))
    end
    return y_test
end

y_test = y_analytical(X_test)

m(X_test)

@test Flux.mse(m(X_test), y_test) < 0.1
