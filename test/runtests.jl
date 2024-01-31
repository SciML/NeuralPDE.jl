using Pkg
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.develop(PackageSpec(path = subpkg_path))
end

@time begin
    if GROUP == "All" || GROUP == "ODEBPINN"
        @time @safetestset "Bpinn ODE solver" begin include("BPINN_Tests.jl") end
    end

    if GROUP == "All" || GROUP == "PDEBPINN"
        @time @safetestset "Bpinn PDE solver" begin include("BPINN_PDE_tests.jl") end
        @time @safetestset "Bpinn PDE invaddloss solver" begin include("BPINN_PDEinvsol_tests.jl") end
    end

    if GROUP == "All" || GROUP == "NNPDE1"
        @time @safetestset "NNPDE" begin include("NNPDE_tests.jl") end
    end
    if GROUP == "All" || GROUP == "NNODE"
        @time @safetestset "NNODE" begin include("NNODE_tests.jl") end
        @time @safetestset "NNODE_tstops" begin include("NNODE_tstops_test.jl") end
        @time @safetestset "NNDAE" begin include("NNDAE_tests.jl") end
    end

    if GROUP == "All" || GROUP == "NNPDE2"
        @time @safetestset "Additional Loss" begin include("additional_loss_tests.jl") end
        @time @safetestset "Direction Function Approximation" begin include("direct_function_tests.jl") end
    end
    if GROUP == "All" || GROUP == "NeuralAdapter"
        @time @safetestset "NeuralAdapter" begin include("neural_adapter_tests.jl") end
    end
    if GROUP == "All" || GROUP == "IntegroDiff"
        @time @safetestset "IntegroDiff" begin include("IDE_tests.jl") end
    end
    if GROUP == "All" || GROUP == "AdaptiveLoss"
        @time @safetestset "AdaptiveLoss" begin include("adaptive_loss_tests.jl") end
    end

    #=
    # Fails because it uses sciml_train
    if GROUP == "All" || GROUP == "NNRODE"
        @time @safetestset "NNRODE" begin include("NNRODE_tests.jl") end
    end
    =#

    if GROUP == "All" || GROUP == "Forward"
        @time @safetestset "Forward" begin include("forward_tests.jl") end
    end
    if GROUP == "All" || GROUP == "Logging"
        dev_subpkg("NeuralPDELogging")
        subpkg_path = joinpath(dirname(@__DIR__), "lib", "NeuralPDELogging")
        Pkg.test(PackageSpec(name = "NeuralPDELogging", path = subpkg_path))
    end
    if !is_APPVEYOR && GROUP == "GPU"
        @safetestset "NNPDE_gpu_Lux" begin include("NNPDE_tests_gpu_Lux.jl") end
    end
end
