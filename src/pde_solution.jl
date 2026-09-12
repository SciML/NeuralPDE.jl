# Wrapping of the OptimizationSolution into the SciMLBase PDE solution interface.

function SciMLBase.PDENoTimeSolution(sol::SciMLBase.PDENoTimeSolution, ::PINNMetadata)
    return sol
end

"""
    PDENoTimeSolution(sol::AbstractOptimizationSolution, md::PINNMetadata)

Wrap the trained network parameters into a `PDENoTimeSolution`. `sol[u(x, t)]` evaluates
the trial function of `u` on the tensor product of the evaluation grids of its arguments
(`sol[x]` returns the grid of `x`), and `sol(x, t; dv = u(x, t))` evaluates it at
arbitrary points. `sol.original_sol` is the underlying `OptimizationSolution`.
"""
function SciMLBase.PDENoTimeSolution(
        sol::SciMLBase.AbstractOptimizationSolution, md::PINNMetadata
    )
    pdesys = md.pdesys
    ivs = collect(get_ivs(pdesys))
    ivgrid = Tuple(collect(md.eval_grid[unwrap(x)]) for x in ivs)
    dvs = collect(get_dvs(pdesys))
    interp = Dict{Any, Any}()
    umap = Dict{Any, Any}()
    T = eltype(sol.u)
    for (dv, net) in zip(dvs, md.networks)
        θ = getu(sol, net.θ)(sol)
        f = trial_function(net, θ)
        interp[dv] = f
        grids = [ivgrid[findfirst(y -> isequal(unwrap(y), unwrap(x)), ivs)] for x in net.args]
        X = _product_matrix(grids)
        umap[dv] = reshape(vec(f(X)), length.(grids)...)
    end
    return SciMLBase.PDENoTimeSolution{
        T, length(dvs), typeof(umap), typeof(md), typeof(sol), typeof(ivgrid),
        typeof(ivs), typeof(dvs), typeof(sol.cache), typeof(sol.alg), typeof(interp),
        typeof(sol.stats),
    }(umap, sol, ivgrid, ivs, dvs, md, sol.cache, sol.alg, interp, sol.retcode, sol.stats)
end

"""
    trial_function(net::TrialNetwork, θ)

Return a callable evaluating the trained network of `net` with parameters `θ`. It
accepts a `d × n` matrix of points (returning a `1 × n` matrix) or `d` scalar coordinates
(returning a number).
"""
function trial_function(net::TrialNetwork, θ)
    wrapper = getdefault(net.NN)
    k = net.output
    T = eltype(θ)
    f(X::AbstractMatrix) = wrapper(X, θ)[k:k, :]
    f(x::AbstractVector{<:Number}) = only(f(reshape(T.(x), :, 1)))
    f(args::Vararg{Number}) = f(collect(args))
    return f
end

function _pinn_call(sol, args...; dv = nothing)
    args = map(enumerate(args)) do (i, arg)
        arg isa Colon ? sol.ivdomain[i] : arg
    end
    if dv === nothing
        length(args) == length(sol.ivs) || throw(
            ArgumentError(
                "Expected $(length(sol.ivs)) arguments for the independent variables, got \
                $(length(args))."
            )
        )
        return map(sol.dvs) do dv
            _pinn_call(sol, args...; dv)
        end
    end
    f = sol.interp[dv]
    dvargs = arguments(unwrap(dv))
    # Either one argument per independent variable of the solution (in `sol.ivs` order) or
    # one per argument of `dv`.
    coords = if length(args) == length(dvargs)
        args
    elseif length(args) == length(sol.ivs)
        is = map(dvargs) do a
            i = findfirst(y -> isequal(unwrap(y), unwrap(a)), sol.ivs)
            i === nothing &&
                throw(ArgumentError("Independent variable $(a) of $(dv) not found."))
            i
        end
        args[is]
    else
        throw(
            ArgumentError(
                "Expected $(length(dvargs)) (arguments of $(dv)) or $(length(sol.ivs)) \
                (independent variables) arguments, got $(length(args))."
            )
        )
    end
    all(c -> c isa Number, coords) && return f(coords...)
    grids = map(c -> c isa Number ? [c] : collect(c), coords)
    X = _product_matrix(grids)
    return reshape(vec(f(X)), length.(grids)...)
end

function (sol::SciMLBase.PDENoTimeSolution{T, N, S, D})(
        args::Vararg{Union{Num, Number, AbstractArray, Colon}}; dv = nothing
    ) where {T, N, S, D <: PINNMetadata}
    return _pinn_call(sol, args...; dv)
end

function _pinn_getindex(A, sym)
    iiv = findfirst(x -> isequal(unwrap(x), unwrap(sym)), A.ivs)
    iiv === nothing || return A.ivdomain[iiv]
    idv = findfirst(x -> isequal(unwrap(x), unwrap(sym)), A.dvs)
    idv === nothing || return A.u[A.dvs[idv]]
    return error("Invalid indexing of solution. $sym not found in solution.")
end

Base.@propagate_inbounds function Base.getindex(
        A::SciMLBase.PDENoTimeSolution{T, N, S, D}, sym::Num
    ) where {T, N, S, D <: PINNMetadata}
    return _pinn_getindex(A, sym)
end

Base.@propagate_inbounds function Base.getindex(
        A::SciMLBase.PDENoTimeSolution{T, N, S, D}, sym::Num, args...
    ) where {T, N, S, D <: PINNMetadata}
    return _pinn_getindex(A, sym)[args...]
end
