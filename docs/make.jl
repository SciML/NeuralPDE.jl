using Documenter, NeuralNetDiffEq

makedocs(
    sitename="NeuralNetDiffEq.jl",
    authors="#",
    clean=true,
    doctest=false,
    modules=[NeuralNetDiffEq],

    format=Documenter.HTML(# analytics = "",
                             assets=["assets/favicon.ico"],
                             canonical="#"),
    pages=[
        "Home" => "index.md",
        "Tutorials" => Any[
            "examples/ode.md",
            "examples/2DPoisson.md",
            "examples/100_HJB.md",
            "examples/kolmogorovbackwards.md",
            "examples/optimal_stopping_american.md",
        ],
        "Neural-Enhanced Solvers" => Any[
            "solvers/ode.md"
            "solvers/pinn.md",
            "solvers/deep_fbsde.md",
            "solvers/kolmogrovbackwards.md",
            "solvers/optimal_stopping.md"
        ]
    ]
)

deploydocs(
   repo="github.com/SciML/NeuralNetDiffEq.jl.git";
   push_preview=true
)
