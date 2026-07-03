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
        # Non-public names still accessed via qualification (verified non-public
        # against the released make-public versions via `Base.ispublic`), by
        # source package:
        #   SciMLBase: AbstractDiscretizationMetadata, __solve, has_analytic,
        #     interp_summary, calculate_solution_errors!
        #   SymbolicUtils: _iszero
        #   Symbolics: variables
        #   ForwardDiff: derivative, jacobian
        #   QuasiMonteCarlo: generate_design_matrices, sample
        #   Base: mapany; Base.Broadcast: dottable
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDiscretizationMetadata, :__solve, :has_analytic,
                :interp_summary, :calculate_solution_errors!,
                :_iszero, :variables, :derivative, :jacobian,
                :generate_design_matrices, :sample, :mapany, :dottable,
            ),
        ),
    ),
)
