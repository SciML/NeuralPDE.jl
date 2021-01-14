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
    # pages=[
    #     "NeuralPDE.jl: Scientific Machine Learning (SciML) for Partial Differential Equations" => "index.md",
    #     "Tutorials" => Any[
    #         "examples/ode.md",
    #         "examples/pinns_example.md",
    #         "examples/100_HJB.md",
    #         "examples/blackscholes.md",
    #         "examples/kolmogorovbackwards.md",
    #         "examples/optimal_stopping_american.md",
    #         "examples/nnrode_example.md",
    #     ],
    #     "Neural-Enhanced PDE Solvers" => Any[
    #         "solvers/ode.md",
    #         "solvers/pinn.md",
    #         "solvers/deep_fbsde.md",
    #         "solvers/kolmogorovbackwards_solver.md",
    #         "solvers/optimal_stopping.md"
    #     ]
    # ]
    pages=[
        "NeuralPDE.jl: Scientific Machine Learning (SciML) for Partial Differential Equations" => "index.md",
        "Symbolic Physics-Informed Neural Network Tutorials" => Any[
            "pinns/pinns_solver.md",
            "pinns/training_strategy.md",
            "pinns/low_level_api.md",#TODO
            "pinns/debugging.md",#TODO
            "examples/pinns_example1.md",
            "examples/pinns_example2.md",
            "examples/pinns_example3.md",
            "examples/pinns_example4.md",
            "examples/pinns_example5.md",
            "examples/pinns_example6.md",
            "examples/pinns_example7.md",
            "examples/pinns_example8.md",
        ],
        "Specialized Neural PDE Tutorials" => Any[
            "solvers/deep_fbsde.md",
            "examples/100_HJB.md",
            "examples/blackscholes.md",

            "solvers/kolmogorovbackwards_solver.md",
            "examples/kolmogorovbackwards.md",

            "solvers/optimal_stopping.md"
            "examples/optimal_stopping_american.md",
        ],
        "Neural Network ODE Solver Tutorials" => Any[
            "solvers/ode.md",
            "examples/ode.md",

            "solvers/nnrode.md",
            "examples/nnrode_example.md",
        ],
        "Manual" => Any[

        ]
    ]
)

deploydocs(
   repo="github.com/SciML/NeuralPDE.jl.git";
   push_preview=true
)
