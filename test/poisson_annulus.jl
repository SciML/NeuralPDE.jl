#
using NeuralPDE, DiffEqFlux, ModelingToolkit, GalacticOptim
import ModelingToolkit: Interval, infimum, supremum
using LinearAlgebra, Plots

r0 = 0.5f0 # inner radius
r1 = 1.0f0 # outer radius

# coordinate transformation
polar(r, θ) = r .* cos.(θ), r .* sin.(θ)

function get_xy_deriv(x,y,r,s)
    Dr = Differential(r)
    Dθ = Differential(θ)

    xr = Dr(x); yr = Dr(y)
    xθ = Dθ(x); yθ = Dθ(y)

    J  = xr*yθ - xθ*yr
    Ji = 1 / J

    rx =  Ji * yθ
    ry = -Ji * xθ
    θx = -Ji * yr
    θy =  Ji * xr

    Dx(v) = rx*Dr(v) + θx*Dθ(v)
    Dy(v) = ry*Dr(v) + θy*Dθ(v)

    Dx, Dy
end

@parameters r θ
@variables u(..)

x, y = polar(r, θ)
Dx, Dy = get_xy_deriv(x,y,r,θ)

Dxx = Dx ∘ Dx
Dyy = Dy ∘ Dy

# PDE
eq = -(Dxx(u(r,θ)) + Dyy(u(r,θ))) ~ 1.0f0

# Boundary conditions
bcs = [
       u(r0, θ) ~ 0.f0,       # Dirichlet, inner
       u(r1, θ) ~ 0.f0,       # Dirichlet, outer
       u(r,0f0) ~ u(r,2f0pi), # Periodic
      ]

domain = [
          r ∈ Interval(r0 ,r1),
          θ ∈ Interval(0f0,2f0pi),
         ]

@named pdesys = PDESystem(eq, bcs, domain, [r, θ], [u])

# Discretization
ndim  = 2
width = 32
depth = 2
act   = σ

NN = Chain(
           Dense(ndim,width,act),
           Chain([Dense(width,width,act) for i in 1:(depth-1)]...),
           Dense(width,1),
          )

initθ    = DiffEqFlux.initial_params(NN)
strategy = QuadratureTraining()
discr    = PhysicsInformedNN(NN,strategy,init_params=initθ)
prob     = discretize(pdesys,discr)

function cb(p,l)
    println("Loss: $l")
    return <(l,1e-5)
end

# training
function train(prob; cb=cb)
    res  = GalacticOptim.solve(prob, ADAM(); cb=cb, maxiters=200)
    prob = remake(prob, u0=res.minimizer)
    res  = GalacticOptim.solve(prob, BFGS(); cb=cb, maxiters=200)
    minimizer = res.minimizer
end

# evaulate solution
function meshplt(x,y,u;a=45,b=60)
    p = plot(x,y,u,legend=false,c=:grays,camera=(a,b))
    p = plot!(x',y',u',legend=false,c=:grays,camera=(a,b))
    return p
end

function eval_sol(phi, minimizer, domain; nsample=50)
    rs,θs=[Array(range(infimum(d.domain),supremum(d.domain),length=nsample))
                                                for d in pdesys.domain]
    
    o  = ones(nsample)
    rs = rs * o'
    θs = o  * θs'

    xs,ys = polar(rs,θs)

    v = zeros(Float32,2,nsample*nsample)
    v[1,:] = rs[:]
    v[2,:] = θs[:]

    upred = phi(v,minimizer)
    upred = reshape(upred,nsample,nsample)

    plt = meshplt(xs,ys,upred)
end

# solution
#min1 = train(prob)
#phi  = discr.phi
#eval_sol(phi, min1, domain)

#@test loss < 1e-4

"""
alternatively solve system in polar coordinates
"""
Dr = Differential(r)
Dθ = Differential(θ)
eq = -1/r * Dr(r * Dr(u(r,θ))) ~ 1.0f0

@named pdesys = PDESystem(eq, bcs, domain, [r, θ], [u])
prob = discretize(pdesys,discr)
min2 = train(prob)
eval_sol(phi, min2, domain)
#
