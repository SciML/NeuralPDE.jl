using NeuralPDE, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(NeuralPDE)
    Aqua.test_ambiguities(NeuralPDE, recursive = false)
    Aqua.test_deps_compat(NeuralPDE)
    Aqua.test_piracies(NeuralPDE)
    Aqua.test_project_extras(NeuralPDE)
    Aqua.test_stale_deps(NeuralPDE)
    Aqua.test_unbound_args(NeuralPDE)
    Aqua.test_undefined_exports(NeuralPDE)
end
