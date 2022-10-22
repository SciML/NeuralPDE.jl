using Documenter, NeuralPDE

ENV["GKSwstype"] = "100"
using Plots

include("pages.jl")

makedocs(sitename = "NeuralPDE.jl",
         authors = "#",
         clean = true,
         doctest = false,
         modules = [NeuralPDE],
         strict = [
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ],
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/NeuralPDE/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/NeuralPDE.jl.git";
           push_preview = true)
