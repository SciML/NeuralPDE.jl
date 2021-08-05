#
"""
 Partitin of Unity Networks with unique affine mappings

 NN: does space partitioning
 Slap on top your favorite function space (cosines)
"""


import Zygote
using ModelingToolkit, Flux, NeuralPDE
using Flux, NNlib, Optim, GalacticOptim, DiffEqFlux
import ModelingToolkit:Interval,infimum,supremum

using LinearAlgebra, Plots, UnPack
using ParameterSchedulers:Scheduler,SinExp
#----------------------------------------------------------------------------#
function affine(x,α,β)
    dimA = length(size(α))
    siz0 = size(x)
    siz1 = Zygote.ignore() do
        return ((1 for i in 1:dimA)...,siz0...)
    end

    x̄ = reshape(x,siz1)
    x̄ = @. α * x̄ + β

    return x̄
end
#----------------------------------------------------------------------------#
struct POUnet

    # partition of unity
    pou

    # unique affine transformation
    α # [ndim,nprt]
    β # [ndim,nprt]

    # coefficient matrix
    C # [ndim,nprt]

end
#--------------------------------------#
function POUnet(nprt,ndim)

    # relu activation function for sharp (piecewise linear)
    # partition functions necessary for representing piecewise
    # continuous functions
    #
    act  = relu
    init = Flux.glorot_uniform
    pou  = Chain(Dense(1   ,nprt,act,initb=init)
                ,Dense(nprt,nprt,act,initb=init)
                ,softmax)
    
    # x -> (α x + β) -> cos(x)
    #
    α = Array{Float32}((0:ndim-1)*ones(1,nprt) )
    β = Array{Float32}(zeros(ndim,nprt))

    C = softmax(init(ndim,nprt);dims=1)

    return POUnet(pou,α,β,C)
end
#--------------------------------------#
Flux.@functor POUnet
Flux.trainable(net::POUnet) = (net.pou,net.α,net.β,net.C)
DiffEqFlux.initial_params(net::POUnet) = Flux.destructure(net)[1]
#--------------------------------------#
function (net::POUnet)(x) # [1,N]

    @unpack pou,α,β,C = net

    N = size(x)[end]

    # partition of unity
    ϕ = pou(x)            # [nprt,N]

    # affine transform
    x̄ = affine(x,α,β)     # [ndim,nprt,N]

    # apply function
    ψ = @. cos(pi*x̄)

    # prod. with coeffs
    f = C .* ψ            # [ndim,nprt,N]
    g = sum(f,dims=1)     # [1,nprt,N]
    h = reshape(g,nprt,N) # [nprt,N]

    # prod. with POU
    h = h .* ϕ 
    y = sum(h,dims=1)     # [1,N]

    return y
end
#----------------------------------------------------------------------------#
function makeplot(phi::POUnet)

    fp = phi(xp)

    plt = plot()
    plt = plot!(title="AFN.jl, nprt=$nprt, ndim=$ndim"
               ,xlabel="x",ylabel="y")
    plt = plot!(ylims=(0,Inf))

    plt = plot!(xp',up',width=3,label="data")
    plt = plot!(xp',fp',width=3,label="prediction")

    # plot partition functions
    par = phi.pou(xp)
    for i=1:size(par,1)
        plt = plot!(xp',par[i,:],width=3
                   ,label=:none,color=:black)
    end

    display(plt)
    return plt
end
#--------------------------------------#
function callback(p,l)

    phi = re(p)
    fp  = phi(xp)
    ep  = up .- fp
    er  = norm(ep,Inf)

    println("Loss: $l, Pointwise er: $er")

    makeplot(phi)

    (l < 1e-6) && return true

    return false
end
#============================================================================#
# driver
#============================================================================#

@parameters x
@variables u(..)
Dx  = Differential(x)
Dxx = Dx ∘ Dx

#--------------------------------------#
# problem setup

function utrue(x)
#   u = @. 2 + sin(3.7*pi*x + pi*0.3)
    u = @. sin(x^2 + 3*x)^2 + exp(x)
#   u = @. 3x^2
#   u = @. 2 + abs(x - 0.5)
#   u = @. 1 + sin(10*x)
#   u = @. sin(pi*x)
    return u
end

# regression
eq = [u(x) ~ utrue(x)]
bc = [u(-1f0) ~ utrue(-1f0)
     ,u( 1f0) ~ utrue( 1f0)]

# Dirichlet BVP
#eq = [-Dxx(u(x)) ~ sin(pi*x)*(pi*pi)]
#bc = [u(-1f0) ~ 0f0
#     ,u( 1f0) ~ 0f0]

domain = [x ∈ Interval(-1f0,1f0)]
#--------------------------------------#

# for evaluation
xp = Array(range(-1,1,length=100))'
up = utrue(xp)

# model
nprt = 1 # partitions
ndim = 4 # banach space dimension
net  = POUnet(nprt,ndim)
_,re = Flux.destructure(net)

# NeuralPDE setup
strategy = QuadratureTraining()
discretization = PhysicsInformedNN(net,strategy)
pdesys = PDESystem(eq,bc,domain,[x],[u])
prob = discretize(pdesys,discretization)

sch  = SinExp(λ0=1e-0,λ1=1e-2,γ=5e-1,period=50)
opt1 = Scheduler(sch,ADAM())
opt2 = BFGS()

res  = GalacticOptim.solve(prob,opt1,cb=callback,maxiters=500)
prob = remake(prob,u0=res.minimizer)
res  = GalacticOptim.solve(prob,opt2,cb=callback,maxiters=300)

min = res.minimizer
phi = re(min)
#phi = discretization.phi # doesn't work for some reason

#--------------------------------------#
nothing
