include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))
include(joinpath(@__DIR__, "..", "helpers", "device_generic.jl"))
using JLArrays

JLArrays.allowscalar(false)
test_device_pde(jl, JLArray)
