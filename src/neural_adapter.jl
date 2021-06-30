function generate_training_sets(domains,dx,eqs,eltypeθ)
    if dx isa Array
        dxs = dx
    else
        dxs = fill(dx,length(domains))
    end
    spans = [infimum(d.domain):dx:supremum(d.domain) for (d,dx) in zip(domains,dxs)]
    train_set = adapt(eltypeθ, hcat(vec(map(points -> collect(points), Iterators.product(spans...)))...))
end


function get_loss_function(loss,initθ,pde_system,strategy::GridTraining)
    eqs = pde_system.eqs
    if !(eqs isa Array)
        eqs = [eqs]
    end
    domains = pde_system.domain
    eltypeθ = eltype(initθ)
    parameterless_type_θ =  DiffEqBase.parameterless_type(initθ)
    dx = strategy.dx
    train_set = generate_training_sets(domains,dx,eqs,eltypeθ)
    get_loss_function(loss,train_set,eltypeθ,parameterless_type_θ,strategy)
end


function neurural_adapter(loss,initθ,pde_system,strategy)
    loss_function__ = get_loss_function(loss,initθ,pde_system,strategy)

    function loss_function_(θ,p)
        loss_function__(θ)
    end
    f_ = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
    prob = GalacticOptim.OptimizationProblem(f_, initθ)
end
