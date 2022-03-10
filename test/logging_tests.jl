import Pkg
Pkg.activate(;temp=true)
Pkg.update()

neuralpde_dir = dirname(abspath(joinpath(@__DIR__, "..")))
@info "neuralpde_directory: $(neuralpde_dir)"
neuralpde = Pkg.PackageSpec(path = neuralpde_dir)
Pkg.develop(neuralpde)

neuralpdelogging_dir = joinpath(neuralpde_dir, "lib", "NeuralPDELogging")
@info "developing NeuralPDELogging subpackage at dir: $(neuralpdelogging_dir)"
neuralpdelogging = Pkg.PackageSpec(path = neuralpdelogging_dir)
Pkg.develop(neuralpdelogging)

@info "building with NeuralPDELogging subpackage"
Pkg.build()
Pkg.precompile()

@info "testing NeuralPDELogging subpackage"
inner_is_CI = haskey(ENV,"CI")
Pkg.test(neuralpdelogging; coverage = inner_is_CI)

