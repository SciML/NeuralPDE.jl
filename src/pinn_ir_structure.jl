"""
    SymbolicPINNIRStructure

A first-class pure symbolic intermediate representation (IR) structure for Physics-Informed
Neural Networks (PINNs) in NeuralPDE.jl.

Preserves residual expressions as uncompiled `BasicSymbolic` objects (`Num`) without compiling
early or approximating derivatives numerically, allowing downstream optimization, symbolic
array grid lowering (`@variables cord_arr[1:D, 1:N]`), and Symbolics codegen pipelines.
"""
struct SymbolicPINNIRStructure{S <: PDESystem}
    pde_residuals::Vector{Num}       # Pure BasicSymbolic PDE residual expressions
    bc_residuals::Vector{Num}        # Pure BasicSymbolic BC residual expressions
    ivs::Vector{Num}                 # Independent variables (e.g. x, y, t)
    dvs::Vector{Num}                 # Dependent variables (e.g. u, v)
    ps::Vector{Num}                  # Symbolic equation parameters
    sys::S                           # Reference to source PDESystem
end

function Base.show(io::IO, ir::SymbolicPINNIRStructure)
    println(io, "SymbolicPINNIRStructure:")
    println(io, "  PDE Residuals (BasicSymbolic): ", length(ir.pde_residuals))
    println(io, "  BC Residuals (BasicSymbolic):  ", length(ir.bc_residuals))
    println(io, "  Independent Variables:        ", ir.ivs)
    println(io, "  Dependent Variables:          ", ir.dvs)
end
