"""
    distill(sol, chain; points = nothing, npoints = 1000, dvs = nothing,
            init_params = nothing, rng = Random.default_rng(), adtype = AutoZygote())
    distill(sols, chain; points = nothing, npoints = 1000, dvs = nothing,
            init_params = nothing, rng = Random.default_rng(), adtype = AutoZygote())

Fit a new network of a different architecture to the evaluations of a trained
`PDENoTimeSolution` and return the `OptimizationProblem` for that regression. Solving
it distills the teacher into the student network (transfer learning across
architectures); warm starts within the same architecture are `remake(prob; u0 = θ)`.

The second form distills several teachers at once (for example one per subdomain of a
domain decomposition) into a single student: each solution is evaluated on its own
points and the targets are concatenated.

`chain` is a Lux layer, or a vector with one Lux layer per distilled dependent
variable. A single layer with several dependent variables is shared, with its `i`-th
output representing the `i`-th distilled variable. `points` is the `d × n` matrix of
training inputs (one column per point, rows in `sol.ivs` order), or a vector with one
such matrix per solution; when omitted, `npoints` inputs per solution are drawn
uniformly over its `ivdomain` with `rng`. `dvs` selects the dependent variables to
distill (a collection, or a single variable) and defaults to all of the (first)
solution's `dvs`; the target rows follow its order. `init_params` is a flat vector, a
Lux parameter `NamedTuple` or a `ComponentArray` (one network), or a vector of those
(one per network). The objective is the mean squared error against
`sol(x...; dv = ...)` on the training inputs, with element type taken from
`init_params` (`Float64` by default).

The returned problem's `u0` is a plain flat vector, so any Optimization.jl optimizer
can solve it; the trained student parameters are `res.u` with the same block layout
(one block per network, in order). To evaluate the student, wrap the blocks back into
their parameter containers:

```julia
template, st = Lux.setup(Xoshiro(0), student)
θ = ComponentArray(res.u, getaxes(ComponentArray(template)))
first(student(reshape([x, y], 2, 1), θ, st))
```

## Example

```julia
teacher = solve(prob, Adam(0.01); maxiters = 1000)
student = Chain(Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 1))
dprob = distill(teacher, student; npoints = 1000, rng = Xoshiro(2))
dres = solve(dprob, Adam(0.01); maxiters = 1000)
```
"""
function distill(
        sol::PDENoTimeSolution, chain; points = nothing, npoints::Int = 1000,
        dvs = nothing, init_params = nothing, rng::AbstractRNG = Random.default_rng(),
        adtype = AutoZygote()
    )
    T = _param_eltype(init_params)
    dvlist = _distill_dvs(sol, dvs)
    X = _distill_points(sol, points, npoints, T, rng)
    Y = _distill_targets(sol, dvlist, X, T)
    return _distill_fit(X, Y, length(dvlist), chain; init_params, rng, adtype)
end

function distill(
        sols::AbstractVector, chain; points = nothing, npoints::Int = 1000,
        dvs = nothing, init_params = nothing, rng::AbstractRNG = Random.default_rng(),
        adtype = AutoZygote()
    )
    isempty(sols) && throw(ArgumentError("`sols` must hold at least one solution."))
    sols = vec(collect(sols))
    d = length(first(sols).ivs)
    all(s -> length(s.ivs) == d, sols) || throw(
        ArgumentError("All distilled solutions must share the independent variables.")
    )
    T = _param_eltype(init_params)
    dvlist = _distill_dvs(first(sols), dvs)
    blocks = points === nothing ? fill(nothing, length(sols)) : vec(collect(points))
    length(blocks) == length(sols) || throw(
        ArgumentError(
            "`points` must hold one matrix per solution, got $(length(blocks)) matrices \
             for $(length(sols)) solutions."
        )
    )
    Xs = map(zip(sols, blocks)) do (s, b)
        _distill_points(s, b, npoints, T, rng)
    end
    Ys = map(zip(sols, Xs)) do (s, X)
        _distill_targets(s, dvlist, X, T)
    end
    return _distill_fit(hcat(Xs...), hcat(Ys...), length(dvlist), chain; init_params, rng, adtype)
end

function _distill_dvs(sol, dvs)
    dvlist = dvs === nothing ? collect(sol.dvs) :
        (dvs isa AbstractArray ? vec(collect(dvs)) : [dvs])
    isempty(dvlist) && throw(ArgumentError("`dvs` must select at least one variable."))
    return dvlist
end

function _distill_points(sol, points, npoints, T, rng)
    d = length(sol.ivs)
    if points === nothing
        npoints >= 1 || throw(ArgumentError("`npoints` must be positive."))
        doms = sol.ivdomain
        lb = T[first(g) for g in doms]
        ub = T[last(g) for g in doms]
        return         rand(rng, T, d, npoints) .* (ub .- lb) .+ lb
    end
    X = Matrix{T}(points)
    size(X, 1) == d || throw(
        ArgumentError(
            "`points` has $(size(X, 1)) rows for $d independent variables; pass a \
             `d × n` matrix with one column per point."
        )
    )
    size(X, 2) >= 1 || throw(ArgumentError("`points` must hold at least one point."))
    return X
end

function _distill_targets(sol, dvlist, X, T)
    d = length(sol.ivs)
    n = size(X, 2)
    Y = Matrix{T}(undef, length(dvlist), n)
    for j in 1:n, (k, dv) in enumerate(dvlist)
        Y[k, j] = sol((X[i, j] for i in 1:d)...; dv = dv)
    end
    return Y
end

function _distill_fit(X, Y, ntargets, chain; init_params, rng, adtype)
    T = eltype(X)
    chains = chain isa AbstractArray ? vec(collect(chain)) : [chain]
    shared = length(chains) == 1
    (shared || length(chains) == ntargets) || throw(
        ArgumentError(
            "Got $(length(chains)) chains for $ntargets distilled variables; pass one \
             shared chain or one chain per variable."
        )
    )
    nets = shared ? [chains[1]] : chains
    psets = map(c -> Lux.setup(rng, c), nets)
    specs = init_params === nothing ? nothing : flat_init_params(init_params)
    specs === nothing || length(specs) == length(nets) || throw(
        ArgumentError("`init_params` must have one entry per network, got $(length(specs)).")
    )
    ax = map(psets) do (ps, _)
        getaxes(ComponentArray(ps))
    end
    len = map(psets) do (ps, _)
        length(ComponentArray(ps))
    end
    θ0 = vcat(map(eachindex(nets)) do i
        tpl = ComponentArray(psets[i][1])
        if specs === nothing
            Vector{T}(tpl)
        else
            v = specs[i]
            length(v) == length(tpl) || throw(
                ArgumentError(
                    "`init_params` entry $i has length $(length(v)); the network has \
                     $(length(tpl)) parameters."
                )
            )
            Vector{T}(copyto!(similar(tpl, T), v))
        end
    end...)
    off = cumsum([0; len])
    need = shared ? ntargets : 1
    for i in eachindex(nets)
        c = ComponentArray(θ0[(off[i] + 1):off[i + 1]], ax[i])
        y_hat, _ = nets[i](X[:, 1:1], c, psets[i][2])
        size(y_hat, 1) == need || throw(
            ArgumentError(
                "Student network $i has $(size(y_hat, 1)) outputs for $need distilled \
                 variables."
            )
        )
    end
    function loss(θ, _)
        rows = map(eachindex(nets)) do i
            c = ComponentArray(θ[(off[i] + 1):off[i + 1]], ax[i])
            y_hat, _ = nets[i](X, c, psets[i][2])
            y_hat
        end
        pred = shared ? only(rows) : vcat(rows...)
        return mean(abs2, pred .- Y)
    end
    return OptimizationProblem(OptimizationFunction(loss, adtype), θ0)
end
