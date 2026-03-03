module GPUUtils

using MLDataDevices: get_device, AbstractGPUDevice

export transform_power_ops, should_apply_gpu_transform

"""
    transform_power_ops(expr::Expr)

Rewrite integer powers in loss function Expr to multiplication chains.
E.g., `(^).(u, 2)` becomes `(*).(u, u)`.

CUDA's `pow` function produces NaN gradients when the base is near zero,
which causes training failures for equations like `u^2` and `Dx(u^3)`.
Using explicit multiplication avoids this issue.

See issue #914.
"""
function transform_power_ops(expr)
    _rewrite_powers(expr)
end

# Check for power operator in both forms: symbol :^ and function object ^
_is_power_op(x) = x === :^ || x isa typeof(^)

# Recursively walk Expr, rewriting integer powers to multiplications
function _rewrite_powers(node)
    node isa Expr || return node

    # Post-order traversal: recurse first, then check patterns
    new_args = map(_rewrite_powers, node.args)
    ex = Expr(node.head, new_args...)

    # Broadcasted power: (^).(base, n)
    if ex.head === :. && length(ex.args) == 2 && _is_power_op(ex.args[1]) &&
       ex.args[2] isa Expr && ex.args[2].head === :tuple
        tup_args = ex.args[2].args
        if length(tup_args) == 2 && _is_positive_int(tup_args[2])
            mul = _matching_mul(ex.args[1])
            return _expand_power_broadcast(mul, tup_args[1], Int(tup_args[2]))
        end
    end

    # Non-broadcasted power: (^)(base, n)
    if ex.head === :call && length(ex.args) == 3 && _is_power_op(ex.args[1]) &&
       _is_positive_int(ex.args[3])
        mul = _matching_mul(ex.args[1])
        return _expand_power_call(mul, ex.args[2], Int(ex.args[3]))
    end

    return ex
end

_is_positive_int(x) = x isa Integer && x >= 0

# Return matching multiplication operator: :* for symbol, * for function
_matching_mul(op) = op isa Symbol ? :* : *

# Transform (^).(base, n) to (*).(base, ..., base)
function _expand_power_broadcast(mul, base, n::Int)
    n == 0 && return 1
    n == 1 && return base
    return Expr(:., mul, Expr(:tuple, fill(base, n)...))
end

# Transform (^)(base, n) to (*)(base, ..., base)
function _expand_power_call(mul, base, n::Int)
    n == 0 && return 1
    n == 1 && return base
    return Expr(:call, mul, fill(base, n)...)
end

"""
    should_apply_gpu_transform(init_params)

Return true if GPU-specific transforms should be applied.

Checks for CUDA device or explicit environment override.
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
