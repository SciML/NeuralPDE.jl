"""
    symbolic_discretize(pdesys::PDESystem, discretization::PhysicsInformedNN)

Lower `pdesys` into a `ModelingToolkit.System` describing the physics-informed neural
network training problem. This runs PDEBase's optimization-system discretization driver
with the hooks implemented below.

Every dependent variable `u(x, t)` is replaced by a `ModelingToolkitNeuralNets`
symbolic network evaluated on a matrix of collocation points, `Differential`s are lowered
with `discretization.derivative`, and every equation becomes one [`ResidualBlock`](@ref)
with its own collocation array parameter. The returned `System` has:

* unknowns: the flat network parameter vectors (and the `PDESystem` parameters when
  `param_estim = true`);
* parameters: the collocation matrices and quadrature weights of each equation, the
  network callables, and the `PDESystem` parameters;
* costs: one mean squared (or quadrature-weighted) residual per PDE, one per boundary
  condition when `boundary_policy == :penalty`, and the `additional_loss` cost;
* constraints: the pointwise boundary residuals when `boundary_policy == :constraints`.

The [`PINNMetadata`](@ref) describing the discretization is stored in the system
metadata under `ModelingToolkitBase.ProblemTypeCtx`.
"""
SciMLBase.symbolic_discretize(::PDESystem, ::PhysicsInformedNN)

"""
    CollocationSpace

The discrete space of a `PhysicsInformedNN` discretization: the variable map of the
system, the trial networks, the symbols standing in for the `PDESystem` parameters and
the boundary-pinned coordinate values.
"""
struct CollocationSpace{V, I, N, P, S, Q, D, T} <: PDEBase.AbstractDiscreteSpace
    varmap::V
    ivs::I
    networks::N
    netmap::P
    ps::S
    param_syms::Q
    pinned::D
    eltype::Type{T}
end

mutable struct PINNState <: PDEBase.AbstractDiscretizationState
    blocks::Vector{ResidualBlock}
end

PDEBase.construct_disc_state(::PhysicsInformedNN) = PINNState(ResidualBlock[])

function PDEBase.construct_discrete_space(
        v::PDEBase.VariableMap, pdesys::PDESystem, disc::PhysicsInformedNN
    )
    T = _param_eltype(disc)
    networks = build_networks(disc, v, T)
    netmap = Dict(net.depvar => net for net in networks)
    ps = _pdesys_params(pdesys)
    param_syms = if disc.param_estim
        Dict{Any, Any}(unwrap(p) => tovar(p) for p in ps)
    else
        Dict{Any, Any}(unwrap(p) => wrap(unwrap(p)) for p in ps)
    end
    pinned = pinned_values(get_bcs(pdesys), v)
    return CollocationSpace(
        v, collect(get_ivs(pdesys)), networks, netmap, ps, param_syms, pinned, T
    )
end

function PDEBase.construct_differential_discretizer(
        pdesys, s::CollocationSpace, disc::PhysicsInformedNN, orders
    )
    return disc.derivative
end

function PDEBase.discretize_equation!(
        state::PINNState, eq::Equation, kind::Symbol, s::CollocationSpace, derivative,
        disc::PhysicsInformedNN
    )
    index = count(b -> b.kind == kind, state.blocks) + 1
    push!(state.blocks, residual_block(eq, kind, index, s, derivative, disc))
    return nothing
end

function PDEBase.generate_metadata(
        s::CollocationSpace, disc::PhysicsInformedNN, pdesys, boundarymap, complexmap, u0
    )
    eval_grid = Dict(
        unwrap(x) => range(s.varmap.intervals[unwrap(x)]...; length = disc.eval_points)
            for x in s.ivs
    )
    return PINNMetadata(pdesys, disc, s.varmap, s.networks, ResidualBlock[], s.ps, eval_grid)
end

function PDEBase.generate_system(
        state::PINNState, s::CollocationSpace, u0, tspan, md::PINNMetadata,
        disc::PhysicsInformedNN; checks = true
    )
    blocks = state.blocks
    append!(md.blocks, blocks)
    T = s.eltype
    costs = Num[]
    constraints = Equation[]
    for b in blocks
        if b.kind == :bc && disc.boundary_policy == :constraints
            push!(constraints, b.residual ~ zeros(T, 1, b.npoints))
        else
            push!(costs, block_cost(b))
        end
    end

    nets = unique_networks(s.networks)
    unknowns_ = Any[net.θ for net in nets]
    params_ = Any[net.NN for net in nets]
    for b in blocks
        b.xs === nothing || push!(params_, b.xs)
        b.w === nothing || push!(params_, b.w)
    end
    psyms = [s.param_syms[unwrap(p)] for p in s.ps]
    if disc.param_estim
        append!(unknowns_, psyms)
    else
        append!(params_, psyms)
    end

    if disc.additional_loss !== nothing
        al = AdditionalLoss(
            disc.additional_loss, Tuple(Symbol(nameof(net.depvar)) for net in nets),
            Tuple(getdefault(net.NN) for net in nets), Tuple(net.output for net in nets)
        )
        alsym = additional_loss_parameter(al)
        push!(params_, alsym)
        push!(costs, alsym((net.θ for net in nets)..., psyms...))
    end

    return System(
        Equation[], unknowns_, params_; costs, constraints, name = nameof(md.pdesys),
        metadata = [ProblemTypeCtx => md], checks
    )
end

_param_eltype(disc::PhysicsInformedNN) = _param_eltype(disc.init_params)
_param_eltype(::Nothing) = Float64
_param_eltype(x::AbstractVector{<:Number}) = float(eltype(x))
_param_eltype(x::AbstractVector) = promote_type(map(_param_eltype, x)...)

function _pdesys_params(pdesys)
    ps = get_ps(pdesys)
    ps isa SciMLBase.NullParameters && return Num[]
    return collect(ps)
end

unique_networks(networks) = unique(net -> unwrap(net.θ), networks)

"""
    build_networks(disc, v, dvs, T)

Create one [`TrialNetwork`](@ref) per dependent variable. A vector of chains gives one
network per dependent variable; a single chain is shared, with its `i`-th output
representing the `i`-th dependent variable.
"""
function build_networks(disc::PhysicsInformedNN, v, T)
    dvs = v.depvar_ops
    ndv = length(dvs)
    chains = disc.chain isa AbstractArray ? disc.chain : fill(disc.chain, ndv)
    length(chains) == ndv || throw(
        ArgumentError(
            "Got $(length(chains)) chains for $(ndv) dependent variables; pass one chain \
            per dependent variable or a single shared chain."
        )
    )
    shared = !(disc.chain isa AbstractArray) && ndv > 1
    init = disc.init_params
    if init !== nothing && !shared
        init = init isa AbstractVector{<:Number} ? [init] : init
        length(init) == ndv || throw(
            ArgumentError("`init_params` must have one entry per network, got $(length(init)).")
        )
    end
    networks = TrialNetwork[]
    shared_net = nothing
    for (i, op) in enumerate(dvs)
        args = v.args[op]
        n_in = length(args)
        if shared
            if shared_net === nothing
                allargs = unique(reduce(vcat, [v.args[d] for d in dvs]))
                length(allargs) == n_in || throw(
                    ArgumentError(
                        "A shared chain requires every dependent variable to have the same \
                        arguments."
                    )
                )
                shared_net = symbolic_network(
                    chains[1], :NN, n_in, ndv, T, init === nothing ? nothing : init, disc.rng
                )
            end
            NN, θ = shared_net
            push!(networks, TrialNetwork(op, args, NN, θ, i, ndv, chains[1]))
        else
            name = nameof(op)
            NN, θ = symbolic_network(
                chains[i], name, n_in, 1, T, init === nothing ? nothing : init[i], disc.rng
            )
            push!(networks, TrialNetwork(op, args, NN, θ, 1, 1, chains[i]))
        end
    end
    return networks
end

function symbolic_network(chain, name, n_in, nout, T, init, rng)
    NN, p = SymbolicNeuralNetwork(;
        chain, n_input = n_in, n_output = nout, rng, eltype = T,
        nn_name = Symbol(:NN_, name), nn_p_name = Symbol(:p_, name)
    )
    np = length(getdefault(p))
    θname = Symbol(:θ_, name)
    θ = only(@variables $θname[1:np])
    init === nothing || length(init) == np || throw(
        ArgumentError(
            "`init_params` for `$(name)` has length $(length(init)); the network has \
            $(np) parameters."
        )
    )
    θ0 = init === nothing ? getdefault(p) : Vector{T}(init)
    θ = setdefault(θ, θ0)
    return NN, θ
end

function additional_loss_parameter(al::AdditionalLoss)
    al_sym = only(@parameters (additional_loss::typeof(al))(..) = al [tunable = false])
    return al_sym
end

"""
    residual_block(eq, kind, index, s::CollocationSpace, derivative, disc)

Lower one equation onto its own collocation set and return the [`ResidualBlock`](@ref).
"""
function residual_block(eq, kind, index, s::CollocationSpace, derivative, disc)
    T = s.eltype
    v = s.varmap
    ex = unwrap(eq.lhs - eq.rhs)
    ivs = free_ivs(ex, s.ivs, v.depvar_ops)
    d = length(ivs)
    ivpos = Int[findfirst(y -> isequal(unwrap(y), unwrap(x)), s.ivs) for x in ivs]
    lb = T[v.intervals[unwrap(x)][1] for x in ivs]
    ub = T[v.intervals[unwrap(x)][2] for x in ivs]
    bpinned = [s.pinned[unwrap(x)] for x in ivs]
    npoints = collocation_count(disc.strategy, kind, ivpos, (lb, ub), bpinned)
    xs = if d == 0
        nothing
    else
        xsname = Symbol(:xs_, kind, index)
        only(@parameters $xsname[1:d, 1:npoints] [tunable = false])
    end
    w = if d > 0 && uses_quadrature_weights(disc.strategy)
        wname = Symbol(:w_, kind, index)
        only(@parameters $wname[1:1, 1:npoints] [tunable = false])
    else
        nothing
    end
    iv_index = Dict(unwrap(x) => i for (i, x) in enumerate(ivs))
    iv_global = Dict(unwrap(x) => i for (i, x) in enumerate(s.ivs))
    ctx = LoweringContext(
        xs === nothing ? nothing : unwrap(xs), iv_index, iv_global, s.netmap,
        s.param_syms, derivative, npoints, T
    )
    residual = lower(ex, ctx, zeros(T, length(s.ivs)))
    return ResidualBlock(eq, kind, ivs, ivpos, (lb, ub), bpinned, xs, w, npoints, residual)
end

"""
    block_cost(block::ResidualBlock)

The scalar cost of a residual block: the mean of the squared pointwise residuals, or
their quadrature-weighted sum when the block carries quadrature weights.
"""
function block_cost(b::ResidualBlock)
    r = b.residual
    if !_isarray(unwrap(r))
        return abs2(r)
    elseif b.w === nothing
        return sum(abs2, r) / b.npoints
    else
        return sum(b.w .* abs2.(r))
    end
end

"""
    discretize(pdesys::PDESystem, discretization::PhysicsInformedNN; kwargs...)

Build the `OptimizationProblem` for training the physics-informed neural network.

`symbolic_discretize` produces the `System`, `mtkcompile` compiles it, the collocation
points are sampled with `discretization.strategy`, and `OptimizationProblem(sys, op;
kwargs...)` generates the objective. All keyword arguments are forwarded to the
`OptimizationProblem` constructor; in particular `adtype` selects the automatic
differentiation backend (default [`default_adtype`](@ref), Zygote) and `weights` scalarizes the costs with
a weighted sum. Parameters of the `PDESystem` without a value in
`pdesys.initial_conditions` must be given through `p`, a collection of `parameter =>
value` pairs.
"""
function SciMLBase.discretize(
        pdesys::PDESystem, disc::PhysicsInformedNN; adtype = default_adtype(), p = (),
        kwargs...
    )
    sys = symbolic_discretize(pdesys, disc)
    md = pinn_metadata(sys)
    csys = mtkcompile(sys)
    PDEBase.add_metadata!(md, csys)
    op = operating_point(md, disc.rng)
    for (k, val) in p
        op[k] = val
    end
    return OptimizationProblem(csys, op; adtype, u0_eltype = _param_eltype(disc), kwargs...)
end

"""
    pinn_metadata(sys)
    pinn_metadata(prob::OptimizationProblem)

Return the [`PINNMetadata`](@ref) of a `System` produced by `symbolic_discretize` or of
the `OptimizationProblem` produced by `discretize`.
"""
pinn_metadata(sys::System) = getmetadata(sys, ProblemTypeCtx, nothing)
pinn_metadata(prob::OptimizationProblem) = pinn_metadata(prob.f.sys)

function operating_point(md::PINNMetadata, rng)
    op = Dict{Any, Any}()
    for net in unique_networks(md.networks)
        op[net.θ] = getdefault(net.θ)
    end
    for b in md.blocks
        b.xs === nothing && continue
        X, W = sample_points(md.disc.strategy, b, rng)
        op[b.xs] = X
        b.w === nothing || (op[b.w] = W)
    end
    ics = initial_conditions(md.pdesys)
    for p in md.ps
        haskey(ics, p) && (op[p] = ics[p])
    end
    return op
end

"""
    resample!(p, md::PINNMetadata; rng = md.disc.rng)

Draw new collocation points for every residual block whose training strategy resamples
(`StochasticTraining` and `QuasiRandomTraining` with `resampling = true`) and store
them in the parameter object `p` of the compiled problem. Use it in a `solve` callback to
recover stochastic training:

```julia
md = pinn_metadata(prob)
cb = (state, loss) -> (resample!(state.p, md); false)
solve(prob, Adam(); callback = cb, maxiters = 1000)
```
"""
function resample!(p, md::PINNMetadata; rng = md.disc.rng)
    sys = md.metadata[]
    sys === nothing && throw(ArgumentError("`resample!` needs metadata from `discretize`."))
    resamples(md.disc.strategy) || return p
    for b in md.blocks
        b.xs === nothing && continue
        X, W = sample_points(md.disc.strategy, b, rng)
        setp(sys, b.xs)(p, X)
        b.w === nothing || setp(sys, b.w)(p, W)
    end
    return p
end
resample!(prob::OptimizationProblem; kwargs...) = resample!(prob.p, pinn_metadata(prob); kwargs...)
