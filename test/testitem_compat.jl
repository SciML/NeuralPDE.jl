"""
Lightweight compatibility shim that replaces ReTestItems.jl macros with
standard-library equivalents while keeping the `@testitem`/`@testsetup`
annotations so that the VSCode Julia extension can still discover tests.

Design:
  * `@testsetup module M ... end`  – evaluates the module at Main scope.
  * `@testitem "name" [tags=...] [setup=[...]] begin ... end`
        – wraps the body in an isolated module (SafeTestsets pattern)
          inside a `@testset`, automatically injecting
          `using Test, NeuralPDE` and `using ..S` for each setup module.
        – when `GROUP` (from runtests.jl) is not "all", skips test items
          whose tags do not match the current group.
"""

"""
    @testsetup module Name ... end

Evaluate the setup-module definition in the current (Main) scope so that
test items can later pull it in with `using ..Name`.
"""
macro testsetup(mod)
    mod.head === :module ||
        error("@testsetup expects a `module ... end` block")
    return esc(mod)
end

"""
    @testitem "name" [tags=[:t1]] [setup=[Mod1, Mod2]] begin ... end

Run the body inside a `@testset` wrapped in a freshly-`@eval`-ed module
that automatically imports `Test`, `NeuralPDE`, and any setup modules.

When `GROUP != "all"`, items whose `tags` do not contain `Symbol(GROUP)` are
skipped (matching ReTestItems' tag-filtering behaviour).
"""
macro testitem(name, args...)
    isempty(args) && error("@testitem requires a begin...end body")
    body = args[end]

    # --- parse optional keyword arguments (tags, setup) ---------------
    _unwrap(x::QuoteNode) = x.value
    _unwrap(x) = x

    setup_modules = Symbol[]
    tags = Symbol[]
    for arg in args[1:(end - 1)]
        if arg isa Expr && arg.head === :(=)
            kw = arg.args[1]
            val = arg.args[2]
            if kw === :tags
                if val isa Expr && val.head === :vect
                    append!(tags, map(_unwrap, val.args))
                else
                    push!(tags, _unwrap(val))
                end
            elseif kw === :setup
                if val isa Expr && val.head === :vect
                    append!(setup_modules, val.args)
                else
                    push!(setup_modules, val)
                end
            else
                @warn "Unknown @testitem keyword: $kw"
            end
        end
    end

    # --- tag-based filtering ------------------------------------------
    tags_tuple = Expr(:tuple, QuoteNode.(tags)...)

    # --- build the module body -----------------------------------------
    setup_usings = [Expr(:using, Expr(:., :., :., s)) for s in setup_modules]

    mod_name = gensym(string("TestItem_", name))

    mod_body = quote
        using Test
        using NeuralPDE
        $(setup_usings...)
        $(body.args...)
    end

    return quote
        let _tags = $tags_tuple,
                _group = isdefined(Main, :GROUP) ? Main.GROUP : "all"

            if _group == "all" || Symbol(_group) in _tags
                @testset $name begin
                    @eval module $(mod_name)
                    $(mod_body.args...)
                    end   # module
                end       # @testset
            else
                @info string(
                    "Skipping test item \"", $name, "\" (tags=", _tags,
                    ", group=", _group, ")"
                )
            end
        end
    end |> esc
end
