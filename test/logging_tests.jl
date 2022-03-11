import Pkg

neuralpde_dir = abspath(joinpath(@__DIR__, ".."))
@info "neuralpde_directory: $(neuralpde_dir)"

neuralpdelogging_dir = abspath(joinpath(neuralpde_dir, "lib", "NeuralPDELogging"))
@info "developing NeuralPDELogging subpackage at dir: $(neuralpdelogging_dir)"
neuralpdelogging = Pkg.PackageSpec(path = neuralpdelogging_dir)
@info "neuralpdelogging: $(string(neuralpdelogging))"

@info "testing NeuralPDELogging subpackage"
inner_is_CI = haskey(ENV,"CI")
@info "inner_is_CI: $(inner_is_CI)"
Pkg.activate(neuralpdelogging_dir)
@info Pkg.project()
Pkg.test(; coverage = inner_is_CI)

