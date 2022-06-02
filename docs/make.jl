using Documenter, NeuralPDE

include("pages.jl")

makedocs(
    sitename="NeuralPDE.jl",
    authors="#",
    clean=true,
    doctest=false,
    modules=[NeuralPDE],

    format=Documenter.HTML(  analytics = "UA-90474609-3",
                             assets=["assets/favicon.ico"],
                             canonical="https://neuralpde.sciml.ai/stable/"),
    pages=pages
)

deploydocs(
   repo="github.com/SciML/NeuralPDE.jl.git";
   push_preview=true
)
