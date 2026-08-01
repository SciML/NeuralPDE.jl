using SciMLTesting, NeuralPDE, Test

# ExplicitImports only sees an extension once its trigger packages are loaded, so
# load every weakdep here to bring NeuralPDEBPINNExt and
# NeuralPDETensorBoardLoggerExt into the set of modules run_qa scans.
using AdvancedHMC, LogDensityProblems, MCMCChains, TensorBoardLogger

function _is_reexported_api(pkg::Module, name::Symbol)
    isdefined(pkg, name) || return false
    value = getfield(pkg, name)
    return try
        parentmodule(value) !== pkg
    catch
        true
    end
end

const REEXPORTED_API = Tuple(
    filter(name -> _is_reexported_api(NeuralPDE, name), SciMLTesting.public_api_names(NeuralPDE))
)

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
    api_docs_kwargs = (;
        rendered = true,
        # NeuralPDE reexports ModelingToolkit and SciMLBase; those names are
        # documented in their owning packages, not in NeuralPDE's API reference.
        ignore = REEXPORTED_API,
        rendered_ignore = REEXPORTED_API,
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
        #   NeuralPDE: logvector, the vector counterpart of the public `logscalar`
        #     logging hook that NeuralPDETensorBoardLoggerExt implements. Promoting it
        #     to public API needs a docstring, a docs entry and a version bump, so it
        #     is a separate change.
        #   AdvancedHMC: make_kernel, the sampler-to-kernel constructor
        #     NeuralPDEBPINNExt needs to turn an HMC/NUTS/HMCDA spec into a kernel.
        #     AdvancedHMC exports no equivalent.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDiscretizationMetadata, :__solve, :has_analytic,
                :interp_summary, :calculate_solution_errors!,
                :_iszero, :variables, :derivative, :jacobian,
                :generate_design_matrices, :sample, :mapany, :dottable,
                :logvector, :make_kernel,
            ),
        ),
        # NeuralPDEBPINNExt is part of NeuralPDE, but ExplicitImports sees it as a
        # separate module, so NeuralPDE's own non-public helpers look like external
        # internals. There is no public spelling for any of these.
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractTrainingStrategy, :BPINNstats, :get_dataset_train_points,
                :merge_strategy_with_loglikelihood_function, :safe_expand,
                :safe_get_device,
            ),
        ),
    ),
)
