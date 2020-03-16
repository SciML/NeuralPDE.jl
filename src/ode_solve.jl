struct NNODE{C,O,P,K} <: NeuralNetDiffEqAlgorithm
    chain::C
    opt::O
    initθ::P
    autodiff::Bool
    kwargs::K
end
function NNODE(chain,opt=Optim.BFGS(),init_params = nothing;autodiff=false,kwargs...)
    if init_params === nothing
        if chain isa FastChain
            initθ = DiffEqFlux.initial_params(chain)
        else
            initθ,re  = Flux.destructure(chain)
        end
    else
        initθ = init_params
    end
    NNODE(chain,opt,initθ,autodiff,kwargs)
end

@nograd append!
function append2(array, input)
    #This function is necessary bc without @nograd, zygote prevents the
    #mutation of arrays. This gets around that *hopefully*
    append!(array, input)
end
@nograd append2
@nograd println


function DiffEqBase.solve(
    prob::DiffEqBase.AbstractODEProblem,
    alg::NeuralNetDiffEqAlgorithm,
    args...;
    dt,
    timeseries_errors = true,
    save_everystep=true,
    adaptive=false,
    abstol = 1f-6,
    verbose = false,
    maxiters = 100,
    do_growing_dt = false
    do_random_kick = true)

    DiffEqBase.isinplace(prob) && error("Only out-of-place methods are allowed!")

    u0 = prob.u0
    tspan = prob.tspan
    f = prob.f
    p = prob.p
    t0 = tspan[1]

    #hidden layer
    chain  = alg.chain
    opt    = alg.opt
    autodiff = alg.autodiff

    #train points generation
    ts = tspan[1]:dt:tspan[2]
    Nt = size(ts)[1] #Number of time steps we will take
    initθ = alg.initθ

    #####DATA STRUCTURES FOR OPTIONS########
    losses = zeros(0) #Here we initialize empty arrays to store
    kicks = zeros(0)  #losses and kicks for analysis and determining
    multiplier = 4.0f0    #future losses and kicks
    multipliers = zeros(0) #this is just so i can plot it at the end to see whats going on.
    counter = 0 #
    ########################################

    if chain isa FastChain
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> u0 + (t-tspan[1])*first(chain(adapt(typeof(θ),[t]),θ))
        else
            phi = (t,θ) -> u0 + (t-tspan[1])*chain(adapt(typeof(θ),[t]),θ)
        end
    else
        _,re  = Flux.destructure(chain)
        #The phi trial solution
        if u0 isa Number
            phi = (t,θ) -> u0 + (t-tspan[1])*first(re(θ)(adapt(typeof(θ),[t])))
        else
            phi = (t,θ) -> u0 + (t-tspan[1])*re(θ)(adapt(typeof(θ),[t]))
        end
    end

    if autodiff
        dfdx = (t,θ) -> ForwardDiff.derivative(t->phi(t,θ),t)
    else
        dfdx = (t,θ) -> (phi(t+sqrt(eps(t)),θ) - phi(t,θ))/sqrt(eps(t))
    end

    function inner_loss(t,θ)
        return sum(abs2,dfdx(t,θ) - f(phi(t,θ),p,t))
    end

    function loss(θ)

        if do_growing_dt
            #=
            To Do!
            #the for t in ts, just rescale t so you have less time steps.
            #Test performance
            =#
        else
            true_loss = sum(abs2,inner_loss(t,θ) for t in ts)
        end

        append2(losses, true_loss)
        counter += 1
        if do_random_kick
            if length(losses) > 6 && counter ≥ 3 && !any(e-> e > losses[end]*1.1, losses[end-5:end])
                #True if there last 5 losses are within 10%
                # and length of the array > 6 so we can do this test
                #We are stuck in a potential well... we need a larger multiplier.
                #Here we'll increase the multiplier by 40%.
                multiplier = multiplier * 1.4f0

                counter = 0 #Reset counter to zero. Count to 3 before increasing multiplier agian.
            end

            rand_kick = convert(Float32, multiplier*rand(1)[1]+1.0)
            append2(kicks, rand_kick)
        else
            rand_kick = 1.0f0
        end

        append2(multipliers, multiplier)
        if verbose
            print("-------------------------------\n")
            print("True_loss: ", true_loss, "\n")
            print("Random Kick: ", rand_kick, "\n")
        end

        if size(losses)[1] > 3 && losses[end-1]/losses[end] > 1.01
            multiplier = 4.0f0 #if we decrease error by 1%, we should reset multiplier
        end

        return true_loss * rand_kick
    end

    cb = function (p,l)
        verbose && println("Current total loss is: $l")
        l < abstol
    end


    res = DiffEqFlux.sciml_train(loss, initθ, opt; cb = cb, maxiters=maxiters, alg.kwargs...)

    #solutions at timepoints
    if u0 isa Number
        u = [first(phi(t,res.minimizer)) for t in ts]
    else
        u = [phi(t,res.minimizer) for t in ts]
    end

    sol = DiffEqBase.build_solution(prob,alg,ts,u,calculate_error = false)
    DiffEqBase.has_analytic(prob.f) && DiffEqBase.calculate_solution_errors!(sol;timeseries_errors=true,dense_errors=false)
    return sol, losses, kicks, multipliers
end #solve


#=
NOTES:
I wonder if we can do something like if the loss is remaining staginant for a while, we apply bigger and bigget kicks? and if it
goes back to where it was, we provide an even bigger kick.

The problem is - I belive that when we increase the multiplier, it is less and less likely to find a smaller answer. So - if we
increase the multiplier, we should first
1. Wait N iterations - maybe 3?
2. If we have decreased our error by some %, maybe 10? then we should rescale the multiplier down to the original value. Try and find
another potential well...
-> This didn't actually work. The code still seems to settle at a true_loss of ~52, despirte growing random_kicks. I wonder if it has
become "too settled"? Is that a thing, by waiting too many iterations, is it harder to get the neural net to find a better value?
-> adding in the reset multiplier seems to get the error down a bit. Right now we're hovering around ~46 at around 40 iterations.
    However, the random kick still seems to be growing arbitrarily high... Maybe 10% is too high of a reduction to reset multiplier
    we're rarely getting over a 1% reduction rate. I think I will try 1% next round, as a multiplier of 78x is way too high.

We can also vary time step to get bigger as loss() gets smaller
=#
