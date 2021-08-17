using Documenter, NeuralPDE

makedocs(
    sitename="NeuralPDE.jl",
    authors="#",
    clean=true,
    doctest=false,
    modules=[NeuralPDE],

    format=Documenter.HTML(# analytics = "",
                             assets=["assets/favicon.ico"],
                             canonical="https://neuralpde.sciml.ai/stable/"),
    pages=[
        "Introduction" => "index.md",

        "Manual on problems" => Any["problems/pinns.md",#TODO
                                    "problems/deep_fbsde.md",#TODO
                                    "problems/kolmogorovbackwards_solver.md",#TODO
                                    "problems/optimal_stopping.md",#TODO
                                    "problems/ode.md",#TODO
                                    "problems/nnrode.md"#TODO
                                    ],

        "Manual on solvers" => Any["solvers/pinns.md",
                                   "solvers/deep_fbsde.md",
                                   "solvers/kolmogorovbackwards_solver.md",
                                   "solvers/optimal_stopping.md",#TODO
                                   "solvers/ode.md",
                                   "solvers/nnrode.md",#TODO
                                   ],

        "Physics-Informed Neural Network Tutorial" => Any[

        #TODO Example
              # 1_dim
              # 2_dim
              # 3_dim?
              # time_depended?
              # system_pde
              # high_order_derivative
              # Integro_differentail
              # parameter_estimation
              # GPU
              # ode
              # aprox_func
              # "low_level.md",
              # "debugging.md",

            "Example" => Any["pinn/poisson.md",
                             "pinn/wave.md",
                             "pinn/diffusion.md",
                             "pinn/2D.md",
                             "pinn/system.md",
                             "pinn/3rd.md",
                             "pinn/low_level.md",
                             "pinn/ks.md",
                             "pinn/fp.md",
                             "pinn/parm_estim.md",
                             "pinn/debugging.md",
                             "pinn/aprox_func.md"
                             ],

            "Idea" => Any["pinn_idea/transfer_learning.md", #TODO
                          "pinn_idea/domain_decomposition.md" #TODO
                          "pinn_idea/adaptive_derivative.md]",#TODO
                          "pinn_idea/adaptive_losses.md" #TODO
                          ],

            "Functionality" => "functionality.md"

                          # remake
                          # neural adapter
                          # additional loss
                          # @register,
                          # callback
                          # float32_64
                          ]
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
        ]
    ]
)

deploydocs(
   repo="github.com/SciML/NeuralPDE.jl.git";
   push_preview=true
)
