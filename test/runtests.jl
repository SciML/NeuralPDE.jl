using Pkg
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

const is_APPVEYOR = Sys.iswindows() && haskey(ENV, "APPVEYOR")

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.develop(PackageSpec(path = subpkg_path))
end

@time begin
    if GROUP == "All" || GROUP == "QA"
        @time @safetestset "Quality Assurance" include("qa.jl")
    end

    if GROUP == "All" || GROUP == "ODEBPINN"
        @time @safetestset "BPINN ODE solver" include("BPINN_Tests.jl")
    end

    if GROUP == "All" || GROUP == "PDEBPINN"
        @time @safetestset "BPINN PDE solver" include("BPINN_PDE_tests.jl")
        @time @safetestset "BPINN PDE invaddloss solver" include("BPINN_PDEinvsol_tests.jl")
    end

    if GROUP == "All" || GROUP == "NNPDE1"
        @time @safetestset "NNPDE" include("NNPDE_tests.jl")
    end

    if GROUP == "All" || GROUP == "NNODE"
        @time @safetestset "NNODE" include("NNODE_tests.jl")
        @time @safetestset "NNODE_tstops" include("NNODE_tstops_test.jl")
        @time @safetestset "NNDAE" include("NNDAE_tests.jl")
    end

    if GROUP == "All" || GROUP == "NNPDE2"
        @time @safetestset "Additional Loss" include("additional_loss_tests.jl")
        @time @safetestset "Direction Function Approximation" include("direct_function_tests.jl")
    end

    if GROUP == "All" || GROUP == "NeuralAdapter"
        @time @safetestset "NeuralAdapter" include("neural_adapter_tests.jl")
    end

    if GROUP == "All" || GROUP == "IntegroDiff"
        @time @safetestset "IntegroDiff" include("IDE_tests.jl")
    end

    if GROUP == "All" || GROUP == "AdaptiveLoss"
        @time @safetestset "AdaptiveLoss" include("adaptive_loss_tests.jl")
    end

    if GROUP == "All" || GROUP == "NNRODE"
        @time @safetestset "NNRODE" include("NNRODE_tests.jl")
    end

    if GROUP == "All" || GROUP == "Forward"
        @time @safetestset "Forward" include("forward_tests.jl")
    end

    if GROUP == "All" || GROUP == "Logging"
        dev_subpkg("NeuralPDELogging")
        subpkg_path = joinpath(dirname(@__DIR__), "lib", "NeuralPDELogging")
        # XXX: problem in TensorBoardLogger that causes error if run with --depwarn=error
        Pkg.test(PackageSpec(; name = "NeuralPDELogging", path = subpkg_path);
            julia_args = ["--depwarn=yes"])
    end

    if !is_APPVEYOR && GROUP == "GPU"
        @safetestset "NNPDE_gpu_Lux" include("NNPDE_tests_gpu_Lux.jl")
    end

    if GROUP == "All" || GROUP == "DGM"
        @time @safetestset "Deep Galerkin solver" include("dgm_test.jl")
    end
end
