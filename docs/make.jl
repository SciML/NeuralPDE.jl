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
            "examples/optimal_stopping.md"

        ],
        "Kolmogorov On GPU" => "kolmogorov_on_gpu.md",
    ]
)

deploydocs(
   repo = "github.com/SciML/NeuralNetDiffEq.jl.git";
   push_preview = true
)
