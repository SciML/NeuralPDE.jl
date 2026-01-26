module GPUUtils

using Symbolics, SymbolicUtils
using MLDataDevices: get_device, AbstractGPUDevice

export transform_power_ops, should_apply_gpu_transform

# Symbolic Expression Transformation

"""
    transform_power_ops(expr)

Transform integer power operations into explicit multiplication chains 
compatible with symbolic differentiation.

This function rewrites expressions of the form `u^n` (where `n` is a positive 
integer) into equivalent multiplication expressions `u * u * ... * u` (n times).
This transformation enables automatic differentiation through the Symbolics.jl
chain rule without requiring special-cased derivative rules for power operations.

Example:
- `u^2` → `u * u`
- `u^3` → `u * u * u`
- `u^4` → `u * u * u * u`
"""
function transform_power_ops(expr)
    count = Ref(0)

    # Extract base expression from ModelingToolkit wrapper if present

    was_num = Symbolics.istype(Symbolics.Num, expr)
    base_expr = was_num ? Symbolics.unwrap(expr) : expr

    transformed = SymbolicUtils.postwalk(base_expr) do node
        # Process SymbolicUtils.Term nodes (symbolic expression terms)
        if node isa SymbolicUtils.Term
            f = node.f
            args = node.args
            # Match power operations: either direct (:^) or call form
            if f === :^ || (f === :call && length(args) >= 2 && args[1] === :^)
                base = args[1]
                exponent = args[2]
                
                # Transform only when exponent is a literal integer or integer-valued number
                if exponent isa Integer || (exponent isa Number && exponent == floor(exponent))
                    n = Int(exponent)
                    count[] += 1
                    
                    if n == 0
                        return 1
                    elseif n == 1
                        return base
                    elseif n == 2
                        # u^2 → u * u
                        return SymbolicUtils.Term(:call, :*, base, base)
                    elseif n == 3
                        # u^3 → u * u * u
                        return SymbolicUtils.Term(:call, :*, base, base, base)
                    else
                        # Unroll arbitrary exponents: u^n → u * u * ... * u (n factors)
                        y = base
                        for i in 2:n
                            y = SymbolicUtils.Term(:call, :*, y, base)
                        end
                        return y
                    end
                end
            end
        
        # Process Expr nodes (unevaluated expressions)
        elseif node isa Expr
            if node.head === :call && length(node.args) >= 3 && node.args[1] === :^
                base = node.args[2]
                exponent = node.args[3]
                
                if exponent isa Integer || (exponent isa Number && exponent == floor(exponent))
                    n = Int(exponent)
                    count[] += 1
                    
                    if n == 0
                        return :(1)
                    elseif n == 1
                        return base
                    elseif n == 2
                        return :($base * $base)
                    elseif n == 3
                        return :($base * $base * $base)
                    else
                        # Expand via nested multiplication Expr nodes
                        expr_mul = base
                        for i in 2:n
                            expr_mul = Expr(:call, :*, expr_mul, base)
                        end
                        return expr_mul
                    end
                end
            end
        end
        
        return node
    end

    # Debug logging
    if count[] > 0 && get(ENV, "NEURALPDE_DEBUG", "0") == "1"
        @info "GPU power transformation: expanded $(count[]) power operations to multiplication chains"
    end

    # Re-attach ModelingToolkit wrapper if the input was wrapped

    return was_num ? Symbolics.Num(transformed) : transformed
end

# GPU Device Detection

"""
    should_apply_gpu_transform(init_params)

Determine whether symbolic power operation transformation should be applied
based on the target computational device.

This function detects if `init_params` corresponds to GPU device parameters.
When GPU device is detected, power operations are expanded into multiplication
chains to enable efficient automatic differentiation on GPU accelerators.

Arguments:
- `init_params`: Model initialization parameters, typically from a Lux neural network

Returns:
- `true` if parameters are allocated on a GPU device
- `false` otherwise, or if `init_params` is `nothing`
"""
function should_apply_gpu_transform(init_params)
    init_params === nothing && return false

    # Allow explicit override via environment variable for development and testing
    if get(ENV, "NEURALPDE_GPU_POWER_REWRITE", "0") == "1"
        return true
    end

    # Detect GPU devices using the MLDataDevices.jl abstraction
    try
        return get_device(init_params) isa AbstractGPUDevice
    catch
        # If device detection fails, default to CPU mode (no transformation)
        return false
    end
end
end # module GPUUtils