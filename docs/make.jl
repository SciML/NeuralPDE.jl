using Documenter, ModelingToolkit, NeuralPDE, SciMLBase

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

ENV["JULIA_DEBUG"] = "Documenter"
ENV["GKSwstype"] = "100"
using Plots

include("pages.jl")

makedocs(
    sitename = "NeuralPDE.jl",
    authors = "#",
    modules = [NeuralPDE],
    clean = true, doctest = true, checkdocs = :exports, linkcheck = true,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/NeuralPDE/stable/"
    ),
    pages = pages
)

deploydocs(repo = "github.com/SciML/NeuralPDE.jl.git"; push_preview = true)
