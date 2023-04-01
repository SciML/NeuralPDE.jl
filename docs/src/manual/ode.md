# ODE-Specialized Physics-Informed Neural Network (PINN) Solver

```@docs
NNODE

additional_loss as an argument for NNODE algorithm:
Any function which computes the additional_loss can be passed as an argument to NNODE algorithm call.

example: 
(u_, t_) = (u_analytical(sol.t), sol.t)
function additional_loss(phi, θ)
    return sum(sum(abs2, [phi(t, θ) for t in t_] .- u_)) / length(u_)
end

alg1 = NeuralPDE.NNODE(chain, opt, strategy=StochasticTraining(1000), additional_loss=additional_loss)

Here we define the additional loss function additional_loss(phi, θ ), the function has two arguments:

phi the trial solution
θ the parameters of neural networks

Note:Refering to above example phi can only take in t as a scalar at a time and θ as parameters of the network.
```
