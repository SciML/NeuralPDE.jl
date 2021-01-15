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
        "Symbolic Physics-Informed Neural Network Tutorials" => Any[
            "examples/pinns_example1.md",
            "examples/pinns_example2.md",
            "examples/pinns_example3.md",
            "examples/pinns_example4.md",
            "examples/pinns_example5.md",
            "examples/pinns_example6.md",
            "examples/pinns_example7.md",
            "examples/pinns_example8.md",
            "examples/pinns_debugging.md",#TODO
        ],
        "Specialized Neural PDE Tutorials" => Any[
            "examples/100_HJB.md",
            "examples/blackscholes.md",
            "examples/kolmogorovbackwards.md",
            "examples/optimal_stopping_american.md",
        ],
        "Specialized Neural ODE Tutorials" => Any[
            "examples/ode.md",
            "examples/nnrode_example.md",
        ],
        "API Documentation" => Any[
            "solvers/pinns.md",#TODO
            "solvers/deep_fbsde.md",
            "solvers/kolmogorovbackwards_solver.md",
            "solvers/optimal_stopping.md",
            "solvers/ode.md",
            "solvers/nnrode.md",#TODO
        ]
    ]
)

deploydocs(
   repo="github.com/SciML/NeuralPDE.jl.git";
   push_preview=true
)
