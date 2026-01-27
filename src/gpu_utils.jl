module GPUUtils

using Symbolics, SymbolicUtils
using MLDataDevices: get_device, AbstractGPUDevice

export transform_power_ops, should_apply_gpu_transform

"""
    transform_power_ops(expr)

Rewrite integer powers (e.g. `u^2`, `u^3`) into explicit multiplication.

This exists to avoid NaNs observed on GPU during training for expressions
like `u(x)^2` and `Dx(u(x)^3)` (see #914). It is not intended as a
performance optimization.
"""

function transform_power_ops(expr)
    count = Ref(0)

    # Extract base expression from ModelingToolkit wrapper if present
    was_num = expr isa Symbolics.Num
    base_expr = was_num ? Symbolics.unwrap(expr) : expr

    transformed = Symbolics.postwalk(base_expr) do node
        # Process BasicSymbolic nodes (symbolic expressions in Symbolics v6+)
        if node isa SymbolicUtils.BasicSymbolic
            op = Symbolics.operation(node)
            args = Symbolics.arguments(node)
            
            # Match power operations
            if op === ^
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
                        # Use SymbolicUtils.term to prevent auto-simplification
                        return SymbolicUtils.term(*, base, base)
                    elseif n == 3
                        return SymbolicUtils.term(*, base, base, base)
                    else
                        # Unroll arbitrary exponents: u^n → u * u * ... * u (n factors)
                        factors = [base for _ in 1:n]
                        return SymbolicUtils.term(*, factors...)
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

"""
    should_apply_gpu_transform(init_params)

Return `true` when GPU-specific symbolic rewrites should be applied

This gates the power-rewriting logic to GPU code paths only (see #914)
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