using Documenter, NeuralPDE

makedocs(
    sitename="NeuralPDE.jl",
    authors="#",
    clean=true,
    doctest=false,
    modules=[NeuralPDE],

    format=Documenter.HTML(# analytics = "",
                             assets=["assets/favicon.ico"],
                             canonical="#"),
    pages=[
        "NeuralPDE.jl: Scientific Machine Learning (SciML) for Partial Differential Equations" => "index.md",
        "Tutorials" => Any[
            "examples/ode.md",
            "examples/pinns_example.md",
            "examples/100_HJB.md",
            "examples/blackscholes.md",
            "examples/kolmogorovbackwards.md",
            "examples/optimal_stopping_american.md",
            "examples/nnrode_example.md",
        ],
        "Neural-Enhanced PDE Solvers" => Any[
            "solvers/ode.md",
            "solvers/pinn.md",
            "solvers/deep_fbsde.md",
            "solvers/kolmogorovbackwards_solver.md",
            "solvers/optimal_stopping.md"
        ]
    ]
)

deploydocs(
   repo="github.com/SciML/NeuralPDE.jl.git";
   push_preview=true
)
