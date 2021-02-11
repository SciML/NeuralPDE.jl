using Pkg
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")

const is_TRAVIS = haskey(ENV,"TRAVIS")

const is_CI = haskey(ENV,"CI")

@time begin
  if GROUP == "All" || GROUP == "NNODE"
      @time @safetestset "NNODE" begin include("NNODE_tests.jl") end
  end
  if !is_APPVEYOR && (GROUP == "All" || GROUP == "NNPDEHan")
      @time @safetestset "NNPDEHan" begin include("NNPDEHan_tests.jl") end
  end
  if GROUP == "All" || GROUP == "NNPDENS"
      @time @safetestset "NNPDENS" begin include("NNPDENS_tests.jl") end
  end
  if GROUP == "All" || GROUP == "NNPDE"
      @time @safetestset "NNPDE" begin include("NNPDE_tests.jl") end
  end
  if GROUP == "All" || GROUP == "NNKOLMOGOROV"
      @time @safetestset "NNKolmogorov" begin include("NNKolmogorov_tests.jl") end
  end
  if GROUP == "All" || GROUP == "NNSTOPPINGTIME"
      @time @safetestset "NNStopping" begin include("Stopping_tests.jl") end
  end
  if GROUP == "All" || GROUP == "NNRODE"
        @time @safetestset "NNRODE" begin include("NNRODE_tests.jl") end
  end
  if GROUP == "All" || GROUP == "NNPKOLMOGOROV"
        @time @safetestset "NNParamKolmogorov" begin include("NNParamKolmogorov_tests.jl") end
  end

  if !is_APPVEYOR && GROUP == "GPU"
     @safetestset "NNPDE_gpu" begin include("NNPDE_tests_gpu.jl") end
 end
end
