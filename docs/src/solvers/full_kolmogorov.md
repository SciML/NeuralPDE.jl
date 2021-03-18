#  Full Kolmogorov PDE Solver
The full Kolmogorov PDE solver obtains the complete solution of a family of γ γ-parameterised Kolmogorov PDE of the form,
![](https://raw.githubusercontent.com/ashutosh-b-b/github-doc-images/master/full_kolmogorov.png)

The  `σ` and `μ` are also parameterised by `γ`.
Thus before we define the neural network solution, we figure out the dimensions required.
Let's say the domain of the PDE is
![](https://raw.githubusercontent.com/ashutosh-b-b/github-doc-images/master/domain_full_kolm.png)
The first dimension is taken by `t` , and the next `d` dimensions are taken by `x`<sup>d</sup> i.e. spatial region. The next dimensions are for `γ` that parameterise the  `σ`, `μ` and `phi` (the initial condition). And thus,
![](https://raw.githubusercontent.com/ashutosh-b-b/github-doc-images/master/param_img.png)
[a<sub>1</sub> | a<sub>2</sub> ....  | a<sub>n</sub>]   represents  horizontal concatenation.
So for defining `γ` we will be defining :
 - `σ_γ` can be atmost d + 1 matrices with dimensions `d x d`  or `nothing`.
 - `μ_γ`  can comprise of either a`d x d` matrix and a matrix of dimensions `d x 1` or both, or nothing.
 - `phi_γ` can be a matrix with dimensions `k x 1`.

We define the prototypes for the above parameters:
- `γ_sigma_prototype`  should be a 3 dimensional matrix with first two dimensions d and d, and the last dimension should be number of matrices (at most d + 1) or it can be `nothing`.
 - `γ_mu_prototype`  should comprise of 2 matrices one with dimensions `d x d `and other with dimensions `d x 1`. So we define `γ_mu_prototype` as `(matrix1 , matrix2)` we put `nothing` in the place if we dont require either of them. Or if we dont want it parameterised, we define,  `γ_mu_prototype` as `nothing`
 - `γ_phi_prototype` should be a matrix with `k x 1` dimensions.

Now lets define a test problem,

We start by defining the dimension `d` of the solution and the prototypes for the problem.
```julia
d = 1
γ_mu_prototype = nothing
γ_sigma_prototype = zeros(d , d , 1)
γ_phi_prototype = nothing
```
And now the neural network solution with the number of dimensions we require at input.
```julia
m = Chain(Dense(3 , 16, elu),Dense(16, 16, elu) , Dense(16 , 5 , elu) , Dense(5 , 1))
```
And then we write the domains of `t` , `x`, `γ_sigma`, `γ_mu` and `γ_phi`.
And thus we use `KolmogorovParamDomain` to pass the domains of   `γ_sigma`, `γ_mu` and `γ_phi` as arguments, respectively.
```julia
tspan = (0.00 , 1.00)
xspan = (0.00 , 3.00)
y_domain = KolmogorovParamDomain((0.00 , 2.00) , (0.00 , 0.00) , (0.00 , 0.00) )
```
The functions `σ(x , γ_sigma)` where  and `μ(x , γ_mu1, γ_mu2)` (`γ_mu1` is the d x d x 1 matrix and `γ_mu2` is a d x 1 x 1 matrix)
```julia
g(x , γ_sigma) = γ_sigma[: , : , 1] #the σ(x , γ_sigma)
f(x , γ_mu_1 ,γ_mu_2 ) = [0.00] # the μ(x , γ_mu1, γ_mu2)
```
And then we define the function `phi(x , γ_phi)` where `γ_phi` is a matrix with `k x 1 x trajectories` . Thus the function should return an `array` with dimensions `1 x trajectories`.
```julia
function phi(x , y_phi)
  x.^2
end
```

We define the `dt` , `dx` and `dy` for the equation,
```julia
dt = 0.01
dx = 0.01
dy = 0.01
```
And then we finally define the optimiser, the algorithms for SDE solution and ensemble simulation.
```julia
opt = Flux.ADAM(1e-2)
sdealg = EM()
ensemblealg = EnsembleThreads()
```
Finally we define the `ParamKolmogorovPDEProblem` ,
```julia
prob = ParamKolmogorovPDEProblem(f , g , phi , xspan , tspan , d  , y_domain  ; Y_sigma_prototype = γ_sigma_prototype)
sol = solve(prob, NNParamKolmogorov(m,opt , sdealg,ensemblealg) , verbose = true, dt = 0.01,
            abstol=1e-10, dx = 0.01 , trajectories = trajectories ,  maxiters = 150 , use_gpu = false)
```
