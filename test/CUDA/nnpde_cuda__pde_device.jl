include(joinpath(@__DIR__, "..", "helpers", "pinn_setup.jl"))
include(joinpath(@__DIR__, "..", "helpers", "device_generic.jl"))
using CUDA

CUDA.allowscalar(false)
test_device_pde(CuArray, CuArray)
