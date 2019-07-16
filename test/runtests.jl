using SafeTestsets

@time begin
@time @safetestset "NNODE" begin include("NNODE_tests.jl") end
@time @safetestset "NNPDEHan" begin include("NNPDEHan_tests.jl") end
end
