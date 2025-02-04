using ReTestItems, InteractiveUtils, Hwloc

@info sprint(versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "all"))

# const RETESTITEMS_NWORKERS = parse(
#     Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 4))))
# const RETESTITEMS_NWORKER_THREADS = parse(Int,
#     get(ENV, "RETESTITEMS_NWORKER_THREADS",
#         string(max(Hwloc.num_virtual_cores() ÷ RETESTITEMS_NWORKERS, 1))))

using BayesianNeuralPDE

@info "Running tests with $(RETESTITEMS_NWORKERS) workers and \
    $(RETESTITEMS_NWORKER_THREADS) threads for group $(GROUP)"

ReTestItems.runtests(BayesianNeuralPDE; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
    nworkers = RETESTITEMS_NWORKERS,
    nworker_threads = RETESTITEMS_NWORKER_THREADS, testitem_timeout = 3600)
