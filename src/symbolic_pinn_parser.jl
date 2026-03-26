# Symbolic PINN Parser MVP

function _build_domain_grids(domains; n_points=20)
    grids = Dict{Any, Vector{Float64}}()
    for dom in domains
        iv = dom.variables
        interval = dom.domain
        lo = DomainSets.leftendpoint(interval)
        hi = DomainSets.rightendpoint(interval)
        grids[iv] = collect(range(Float64(lo), Float64(hi), length=n_points))
    end
    return grids
end

function _replace_dv_calls(expr, dvs, ivs, nn_out_sym)
    dv_op_names = [string(Symbolics.operation(Symbolics.unwrap(dv))) for dv in dvs]
    expr_unwrapped = Symbolics.unwrap(expr)

    pw = SymbolicUtils.Postwalk(x -> begin
        if SymbolicUtils.istree(x)
            op = Symbolics.operation(x)
            op_name = string(op)
            idx = findfirst(==(op_name), dv_op_names)
            if idx !== nothing
                args = collect(Symbolics.arguments(x))
                coord_map = Dict{Any, Any}()
                for i in 1:min(length(ivs), length(args))
                    coord_map[ivs[i]] = args[i]
                end
                nn_at_args = substitute(nn_out_sym[idx], coord_map)
                return Symbolics.unwrap(nn_at_args)
            end
        end
        return x
    end)

    return Num(pw(expr_unwrapped))
end

function _build_compact_symbolic_loss(eqs, bcs, dvs, ivs; n_points=20, bc_weight=10.0)
    iv_list = join(string.(ivs), ", ")

    pde_terms = String[]
    for eq in eqs
        res_str = string(eq.lhs - eq.rhs)
        for (j, dv) in enumerate(dvs)
            dv_op = split(string(dv), "(")[1]
            call_pat = Regex("\\b" * dv_op * "\\([^\\)]*\\)")
            res_str = replace(res_str, call_pat => "NN_$(j)($(iv_list))")
        end
        push!(pde_terms, "(" * res_str * ")^2")
    end

    bc_terms = String[]
    for bc in bcs
        res_bc_str = string(bc.lhs - bc.rhs)
        for (j, dv) in enumerate(dvs)
            dv_op = split(string(dv), "(")[1]
            call_pat = Regex("\\b" * dv_op * "\\([^\\)]*\\)")
            res_bc_str = replace(res_bc_str, call_pat => "NN_$(j)($(iv_list))")
        end
        push!(bc_terms, "(" * res_bc_str * ")^2")
    end

    io = IOBuffer()
    println(io, "# Compact symbolic PINN loss template")
    println(io, "# NN_j(...) are registered neural outputs (not expanded layer-by-layer).")
    println(io)

    println(io, "PDE residual terms:")
    if isempty(pde_terms)
        println(io, "0")
    else
        println(io, join(pde_terms, " + "))
    end
    println(io)

    println(io, "Boundary residual terms:")
    if isempty(bc_terms)
        println(io, "0")
    else
        println(io, join(bc_terms, " + "))
    end
    println(io)

    println(io, "Loss template:")
    println(
        io,
        "sum_over_" * string(n_points) * "_point_grid(PDE residual terms) + " *
        string(bc_weight) * " * (Boundary residual terms)"
    )

    return String(take!(io))
end

function build_pinn_loss(
    pde_system,
    chain = nothing;
    width=16,
    depth=2,
    activation=tanh,
    n_points=20,
    bc_weight=10.0,
    show_symbolic_expression=true,
    symbolic_expression_path="pinn_symbolic_expression.txt",
    symbolic_expression_style=:compact,
    rng=Random.default_rng()
)
    eqs = collect(pde_system.eqs)
    bcs = collect(pde_system.bcs)
    dvs = collect(pde_system.dvs)
    ivs = collect(pde_system.ivs)
    doms = collect(pde_system.domain)

    if chain === nothing
        # We manually construct a fully connected network as fallback
        # If ModelingToolkitNeuralNets is not available we could build manually via Lux
        chain = Lux.Chain(Lux.Dense(length(ivs), width, activation), Lux.Dense(width, length(dvs)))
    end

    p_init, st = Lux.setup(rng, chain)
    p_ca = ComponentArray(p_init)
    @variables p_sym[1:length(p_ca)]
    p_sym_ca = ComponentArray(p_sym, getaxes(p_ca))

    nn_out_sym, _ = Lux.apply(chain, ivs, p_sym_ca, st)

    grids = _build_domain_grids(doms; n_points=n_points)
    
    # 1. Compile PDE residuals into functions
    compiled_res_funcs = []
    for eq in eqs
        res = eq.lhs - eq.rhs
        res = _replace_dv_calls(res, dvs, ivs, nn_out_sym)
        res = expand_derivatives(res)

        # Compile to a function taking (p, ivs...)
        bf = Symbolics.build_function(res, p_sym, ivs..., expression=Val(false))
        push!(compiled_res_funcs, bf isa Tuple ? bf[1] : bf)
    end

    # 2. Compile Boundary Condition residuals
    compiled_bc_funcs = []
    for bc in bcs
        res_bc = bc.lhs - bc.rhs
        res_bc = _replace_dv_calls(res_bc, dvs, ivs, nn_out_sym)
        res_bc = expand_derivatives(res_bc)
        
        # Evaluate boundaries across the grid
        bf_bc = Symbolics.build_function(res_bc, p_sym, ivs..., expression=Val(false))
        push!(compiled_bc_funcs, bf_bc isa Tuple ? bf_bc[1] : bf_bc)
    end
    
    # Grid arrays
    iv_vecs = [grids[iv] for iv in ivs]

    if show_symbolic_expression
        output_path = isabspath(symbolic_expression_path) ? symbolic_expression_path : joinpath(pwd(), symbolic_expression_path)

        expr_text = if symbolic_expression_style == :expanded
            "Expanded symbolic expression tracking full grid is disabled in Lazy Grid Sum mode."
        elseif symbolic_expression_style == :compact
            _build_compact_symbolic_loss(eqs, bcs, dvs, ivs; n_points=n_points, bc_weight=bc_weight)
        else
            error("symbolic_expression_style must be :compact or :expanded")
        end

        open(output_path, "w") do io
            print(io, expr_text)
        end
        println("Symbolic PINN loss expression (" * String(symbolic_expression_style) * ") saved to: " * output_path)
    end

    loss_func = (p) -> begin
        loss = zero(eltype(p))
        
        # PDE Residuals over grid
        for cpt in Iterators.product(iv_vecs...)
            for rf in compiled_res_funcs
                r = rf(p, cpt...)
                loss += r^2
            end
        end
        
        # Boundary Conditions over the grid
        for cpt in Iterators.product(iv_vecs...)
            for bcf in compiled_bc_funcs
                r_bc = bcf(p, cpt...)
                loss += bc_weight * r_bc^2
            end
        end
        
        return loss
    end

    return loss_func, p_ca, chain, st
end
