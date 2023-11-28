using Documenter, NeuralPDE

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

ENV["GKSwstype"] = "100"
ENV["JULIA_DEBUG"] = "Documenter"
using Plots

include("pages.jl")

makedocs(sitename = "NeuralPDE.jl",
         authors = "#",
         modules = [NeuralPDE],
         clean = true, doctest = false, linkcheck = true,
         warnonly = [:missing_docs, :example_block],
         format = Documenter.HTML(assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/NeuralPDE/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/NeuralPDE.jl.git";
           push_preview = true)
