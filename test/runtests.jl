using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")
const is_TRAVIS = haskey(ENV,"TRAVIS")

@time begin
if GROUP == "All" || GROUP == "Test1"
    @time @safetestset "NNODE" begin include("NNODE_tests.jl") end
    @time @safetestset "NNPDEHan" begin include("NNPDEHan_tests.jl") end
end
if GROUP == "All" || GROUP == "Test2"
    @time @safetestset "NNPDENS" begin include("NNPDENS_tests.jl") end
end
end
