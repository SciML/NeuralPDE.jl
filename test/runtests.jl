using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")

const is_TRAVIS = haskey(ENV,"TRAVIS")


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
  if GROUP == "All" || GROUP == "NNKOLMOGOROV"
      @time @safetestset "NNKolmogorov" begin include("NNKolmogorov_tests.jl") end
  end
  if GROUP == "All" || GROUP == "NNSTOPPINGTIME"
      @time @safetestset "NNStopping" begin include("Stopping_tests.jl") end
  end

end
