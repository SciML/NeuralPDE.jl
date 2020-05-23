using Documenter, NeuralNetDiffEq

makedocs(
    sitename = "NeuralNetDiffEq.jl",
    authors="#",
    clean = true,
    doctest = false,
    modules = [NeuralNetDiffEq],

    format = Documenter.HTML(#analytics = "",
                             assets = ["assets/favicon.ico"],
                             canonical="#"),
    pages=[
        "Home" => "index.md",
        "Tutorials" => Any[
            "examples/Relationship_between_SDE_and_PDE.md"
        ],
    ]
)

deploydocs(
   repo = "github.com/SciML/NeuralNetDiffEq.jl.git";
   push_preview = true
)
