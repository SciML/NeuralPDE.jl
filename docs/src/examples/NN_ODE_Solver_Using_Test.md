## Solving ODEs with Neural Networks

Now let's get to our first true SciML application: solving Ordinary Differential Equations with Neural Networks.

We will handle the initial condition. One simple way to do this is to add an initial condition term to the cost function. While that would work, it can be more efficient to encode the initial condition into the function itself so that it's trivially satisfied for any possible set of parameters. For example, instead of directly using a neural network, we can use:

```math
g(t) = u_0 + tNN(t)
```

as our solution. Notice that this will always satisfy the initial condition, so if we train this to satisfy the derivative function then it will automatically be a solution to the derivative function.

## Coding Up the Method

Now let's implement this method with `Flux`, and `NeuralNetDiffEq`. Let's define a neural network to be the `NN(t)` above. To make the problem easier, let's look at the ODE:

```math
u' = \cos 2\pi t
```

and approximate it with the neural network from a scalar to a scalar:

```julia
using Test, Flux, Optim
println("NNODE Tests")
```
**Output**: `NNODE Tests`

Now, we will generate some random seeds:

```julia
using DiffEqDevTools , NeuralNetDiffEq
using Random
Random.seed!(100)
```
**Output**: `MersenneTwister(UInt32[0x00000064], Random.DSFMT.DSFMT_state(Int32[-2036630343, 1072818225, 1299231502, 1073154435, 1563612565, 1073206618, 176198161, 1073683625, 381415896, 1073699088  …  163992627, 1073241259, 385818456, 1072878963, 399273729, 595433664, 390891112, 1704156657, 382, 0]), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  …  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], UInt128[0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000  …  0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000, 0x00000000000000000000000000000000], 1002, 0)`

Now, we will run a solve on Scalars. So, it will calculate loss and Interpolate 1st order Linear and Instead of directly approximating the neural network, we will use the transformed equation :

```julia
linear = (u,p,t) -> cos(2pi*t)
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(linear, u0 ,tspan)
chain = Flux.Chain(Dense(1,5,σ),Dense(5,1))
opt = Flux.ADAM(0.1, (0.9, 0.95))
sol = solve(prob, NeuralNetDiffEq.NNODE(chain,opt), dt=1/20f0, verbose = true,
            abstol=1e-10, maxiters = 200)
```

`Current loss is: 0.0005181264140602735`

**Output**: Interpolation of 1st order Lnear:
```
retcode: Default
Interpolation: 1st order linear
t: 0.0f0:0.05f0:1.0f0
u: 21-element Array{Float32,1}:
  0.0
  0.050792348
  0.09371349
  0.12611231
  0.14565676
  0.15073816
  0.14086685
  0.116954416
  0.081379764
  0.037780646
 -0.009404998
 -0.055532973
 -0.096331164
 -0.12832798
 -0.14906931
 -0.15713885
 -0.15203655
 -0.13398945
 -0.10374507
 -0.062384833
 -0.011173807
 ```

 Now, we will apply `Broyden–Fletcher–Goldfarb–Shanno (BFGS)`: is an iterative method for solving unconstrained nonlinear optimization, to minimize the loss:

 ```julia
 opt = BFGS()
sol = solve(prob, NeuralNetDiffEq.NNODE(chain,opt), dt=1/20f0, verbose = true,
            abstol=1e-10, maxiters = 200)
```

So, now `Current loss is: 0.00020640521549557017` and Interpolation of 1st Order Linear:
```
retcode: Default
Interpolation: 1st order linear
t: 0.0f0:0.05f0:1.0f0
u: 21-element Array{Float32,1}:
  0.0
  0.047355097
  0.09067219
  0.12655868
  0.15135364
  0.16174506
  0.15560715
  0.13281216
  0.09565356
  0.048604924
 -0.0025589764
 -0.051927865
 -0.094458
 -0.12653078
 -0.14604165
 -0.15217933
 -0.14507604
 -0.12546553
 -0.0944093
 -0.053108547
 -0.0027823448
 ```

 Now, we will run a solver on Vectors to calculate and interplate 1st order linear :

 ```julia
 linear = (u,p,t) -> [cos(2pi*t)]
tspan = (0.0f0, 1.0f0)
u0 = [0.0f0]
prob = ODEProblem(linear, u0 ,tspan)
chain = Flux.Chain(Dense(1,5,σ),Dense(5,1))
opt = BFGS()
sol = solve(prob, NeuralNetDiffEq.NNODE(chain,opt), dt=1/20f0, abstol=1e-10,
            verbose = true, maxiters=200)
```

`Current loss is: 0.0001529127732408631`

Output: 
```
retcode: Default
Interpolation: 1st order linear
t: 0.0f0:0.05f0:1.0f0
u: 21-element Array{Array{Float32,1},1}:
 [0.0]
 [0.046757717]
 [0.08994444]
 [0.12591833]
 [0.15083086]
 [0.1612803]
 [0.1551392]
 [0.13231088]
 [0.09509164]
 [0.04791297]
 [-0.0034959316]
 [-0.053215463]
 [-0.096117824]
 [-0.12845704]
 [-0.14801918]
 [-0.15395197]
 [-0.14643712]
 [-0.12634109]
 [-0.094912484]
 [-0.053561118]
 [-0.0037125945]
 ```

 ## Example 1:

 ```julia
 linear = (u,p,t) -> @. t^3 + 2*t + (t^2)*((1+3*(t^2))/(1+t+(t^3))) - u*(t + ((1+3*(t^2))/(1+t+t^3)))
linear_analytic = (u0,p,t) -> [exp(-(t^2)/2)/(1+t+t^3) + t^2]
prob = ODEProblem(ODEFunction(linear,analytic=linear_analytic),[1f0],(0.0f0,1.0f0))
chain = Flux.Chain(Dense(1,128,σ),Dense(128,1))
opt = ADAM(0.01)
sol  = solve(prob,NeuralNetDiffEq.NNODE(chain,opt),verbose = true, dt=1/5f0, maxiters=200)
@test sol.errors[:l2] < 0.5
```

**Output**: `Test Passed`

```
#=
dts = 1f0 ./ 2f0 .^ (6:-1:2)
sim = test_convergence(dts, prob, NeuralNetDiffEq.NNODE(chain, opt))
@test abs(sim.𝒪est[:l2]) < 0.1
@test minimum(sim.errors[:l2]) < 0.5
=#
```

## Example: 2

```julia
linear = (u,p,t) -> -u/5 + exp(-t/5).*cos(t)
linear_analytic = (u0,p,t) ->  exp(-t/5)*(u0 + sin(t))
prob = ODEProblem(ODEFunction(linear,analytic=linear_analytic),0.0f0,(0.0f0,1.0f0))
chain = Flux.Chain(Dense(1,5,σ),Dense(5,1))
opt = ADAM(0.01)
sol  = solve(prob,NeuralNetDiffEq.NNODE(chain,opt),verbose = true, dt=1/5f0)
@test sol.errors[:l2] < 0.5
```

**Output**: `Test Passed`

```
#=
dts = 1f0 ./ 2f0 .^ (6:-1:2)
sim = test_convergence(dts, prob, NeuralNetDiffEq.NNODE(chain, opt))
@test abs(sim.𝒪est[:l2]) < 0.5
@test minimum(sim.errors[:l2]) < 0.1
=#
```
