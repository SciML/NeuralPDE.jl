using Test, InteractiveUtils

@info sprint(versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "all"))

# Ensure relative includes (e.g., burger_reference_data.jl inside dgm_tests.jl)
# resolve correctly when test bodies are @eval-ed into fresh modules.
cd(@__DIR__)

include("testitem_compat.jl")

using NeuralPDE

@info "Running tests for group $(GROUP)"

if GROUP == "all" || GROUP == "qa"
    include("qa_tests.jl")
end

if GROUP == "all" || GROUP == "odebpinn"
    include("BPINN_tests.jl")
end

if GROUP == "all" || GROUP == "pdebpinn"
    include("BPINN_PDE_tests.jl")
end

if GROUP == "all" || GROUP == "nnsde"
    include("NN_SDE_tests.jl")
end

if GROUP == "all" || GROUP == "nnpde1"
    include("NNPDE_tests.jl")
end

if GROUP == "all" || GROUP == "nnpde2"
    include("direct_function_tests.jl")
    include("additional_loss_tests.jl")
end

if GROUP == "all" || GROUP == "adaptiveloss"
    include("adaptive_loss_tests.jl")
end

if GROUP == "all" || GROUP == "forward"
    include("forward_tests.jl")
end

if GROUP == "all" || GROUP == "dgm"
    include("dgm_tests.jl")
end

if GROUP == "all" || GROUP == "nnode"
    include("NNODE_tests.jl")
    include("NNDAE_tests.jl")
end

if GROUP == "all" || GROUP == "pinoode"
    include("PINO_ode_tests.jl")
end

if GROUP == "all" || GROUP == "neuraladapter"
    include("neural_adapter_tests.jl")
end

if GROUP == "all" || GROUP == "integrodiff"
    include("IDE_tests.jl")
end
