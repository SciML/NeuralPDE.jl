using SciMLTesting, NeuralPDE, Test

run_qa(
    NeuralPDE;
    explicit_imports = true,
    # undefined_exports: ModelingToolkit exports AbstractDynamicOptProblem but does
    # not define it; re-exported via @reexport (upstream ModelingToolkit issue).
    # persistent_tasks: SymbolicsPreallocationToolsExt has __precompile__(false),
    # which trips Aqua's precompilation check (upstream Symbolics.jl issue).
    aqua_kwargs = (;
        undefined_exports = false,
        persistent_tasks = false,
    ),
    # ambiguities: PINOODE's `PDETimeSeriesSolution{...,<:PINOODEMetadata}(p, t)`
    # callable is ambiguous with the RecursiveArrayTools/SciMLBase
    # `(t, ::Type{deriv})` interpolation methods (was a hard red on master before
    # this conversion). Tracked in SciML/NeuralPDE.jl#1079; remove when fixed.
    aqua_broken = (:ambiguities,),
    ei_kwargs = (;
        # NeuralPDE @reexport's ModelingToolkit for downstream convenience, which
        # leaks the `ModelingToolkitBase` module name as an implicit import.
        no_implicit_imports = (; skip = (NeuralPDE, Base, Core, ModelingToolkit)),
        # get_dvs/get_ivs are owned by ModelingToolkitBase but accessed via the
        # @reexport'd ModelingToolkit (which re-exports them).
        all_qualified_accesses_via_owners = (; ignore = (:get_ivs, :get_dvs)),
        # Non-public names accessed via qualification. These go public as their base
        # libraries declare them `public`; tracked by source package:
        #   SciMLBase: AbstractDAEAlgorithm, AbstractDAEProblem,
        #     AbstractDiscretizationMetadata, AbstractIntegralAlgorithm,
        #     AbstractODEAlgorithm, AbstractODEProblem, AbstractSDEProblem,
        #     NullParameters, Success, __solve, allowscomplex, build_solution,
        #     calculate_solution_errors!, has_analytic, interp_summary
        #   SymbolicUtils: BasicSymbolic, _iszero, unwrap
        #   Symbolics: value, variables
        #   ModelingToolkit (owned by ModelingToolkitBase): get_dvs, get_ivs
        #   LuxCore: initialparameters, initialstates, parameterlength, setup
        #   Optimisers: setup, update!
        #   ForwardDiff: derivative, jacobian
        #   QuasiMonteCarlo: generate_design_matrices, sample
        #   RuntimeGeneratedFunctions: init
        #   Adapt: adapt_storage; ArrayInterface: allowed_getindex
        #   Random: default_rng; Base: Fix2, mapany; Base.Broadcast: dottable
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDAEAlgorithm, :AbstractDAEProblem,
                :AbstractDiscretizationMetadata, :AbstractIntegralAlgorithm,
                :AbstractODEAlgorithm, :AbstractODEProblem, :AbstractSDEProblem,
                :NullParameters, :Success, :__solve, :allowscomplex, :build_solution,
                :calculate_solution_errors!, :has_analytic, :interp_summary,
                :BasicSymbolic, :_iszero, :unwrap, :value, :variables,
                :get_dvs, :get_ivs,
                :initialparameters, :initialstates, :parameterlength, :setup,
                :update!, :derivative, :jacobian,
                :generate_design_matrices, :sample, :init,
                :adapt_storage, :allowed_getindex, :default_rng,
                :Fix2, :mapany, :dottable,
            ),
        ),
        # LuxCore.{initialparameters,initialstates,parameterlength} are imported to
        # *extend* their methods (dgm.jl etc.) — dropping them breaks downstream
        # method resolution, so they stay imported despite being non-public in
        # LuxCore. recursive_eltype is non-public in Lux; unwrap non-public in
        # SymbolicUtils. All go public as those libs declare them.
        all_explicit_imports_are_public = (;
            ignore = (
                :initialparameters, :initialstates, :parameterlength,
                :recursive_eltype, :unwrap,
            ),
        ),
    ),
)
