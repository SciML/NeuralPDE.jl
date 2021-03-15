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
        "Physics-Informed Neural Network Tutorials" => Any[
            "pinn/poisson.md",
            "pinn/wave.md",
            "pinn/2D.md",
            "pinn/system.md",
            "pinn/3rd.md",
            "pinn/low_level.md",
            "pinn/ks.md",
            "pinn/fp.md",
            "pinn/debugging.md",
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
            "solvers/pinns.md",
            "solvers/deep_fbsde.md",
            "solvers/kolmogorovbackwards_solver.md",
            "solvers/optimal_stopping.md",#TODO
            "solvers/ode.md",
            "solvers/nnrode.md",#TODO
        ]
    ]
)

deploydocs(
   repo="github.com/SciML/NeuralPDE.jl.git";
   push_preview=true
)
