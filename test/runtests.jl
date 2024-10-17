using Pkg, SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

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

    if GROUP == "All" || GROUP == "PINOODE"
        @time @safetestset "pino ode" begin
            include("PINO_ode_tests.jl")
        end
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

    if GROUP == "All" || GROUP == "Forward"
        @time @safetestset "Forward" include("forward_tests.jl")
    end

    if GROUP == "All" || GROUP == "Logging"
        @testset for log_setting in ["NoImport", "ImportNoUse", "ImportUse"]
            ENV["LOG_SETTING"] = log_setting
            @time @safetestset "Logging" include("logging_tests.jl")
        end
    end

    if GROUP == "CUDA"
        @safetestset "NNPDE_gpu_Lux" include("NNPDE_tests_gpu_Lux.jl")
    end

    if GROUP == "All" || GROUP == "DGM"
        @time @safetestset "Deep Galerkin solver" include("dgm_test.jl")
    end
end
