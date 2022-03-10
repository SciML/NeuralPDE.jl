using Pkg
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")

const is_TRAVIS = haskey(ENV,"TRAVIS")

is_CI = haskey(ENV,"CI")


@time begin
  if GROUP == "All" || GROUP == "Logging"
      @time @safetestset "AdaptiveLossLogNoImport" begin 
        using Pkg
        neuralpde_dir = dirname(abspath(joinpath(@__DIR__, "..", "..", "..")))
        @info "loading neuralpde package at : $(neuralpde_dir)"
        neuralpde = Pkg.PackageSpec(path = neuralpde_dir)
        Pkg.develop(neuralpde)
        @info "making sure that there are no logs without having imported NeuralPDELogging"
        ENV["LOG_SETTING"] = "NoImport"
        include("adaptive_loss_log_tests.jl") 
      end
      @time @safetestset "AdaptiveLossLogImportNoUse" begin 
        using Pkg
        neuralpde_dir = dirname(abspath(joinpath(@__DIR__, "..", "..", "..")))
        @info "loading neuralpde package at : $(neuralpde_dir)"
        neuralpde = Pkg.PackageSpec(path = neuralpde_dir)
        Pkg.develop(neuralpde)
        @info "making sure that there are still no logs now that we have imported NeuralPDELogging"
        ENV["LOG_SETTING"] = "ImportNoUse"
        include("adaptive_loss_log_tests.jl") 
      end
      @time @safetestset "AdaptiveLossLogImportUse" begin 
        using Pkg
        neuralpde_dir = dirname(abspath(joinpath(@__DIR__, "..", "..", "..")))
        @info "loading neuralpde package at : $(neuralpde_dir)"
        neuralpde = Pkg.PackageSpec(path = neuralpde_dir)
        Pkg.develop(neuralpde)
        ENV["LOG_SETTING"] = "ImportUse"
        @info "making sure that logs are generated now if we use a logger"
        include("adaptive_loss_log_tests.jl") 
      end
  end
end
