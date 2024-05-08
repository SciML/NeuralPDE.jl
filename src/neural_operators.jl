abstract type NeuralOperator <: Lux.AbstractExplicitLayer end

"""
DeepONet(branch,trunk)
"""

"""
    DeepONet(branch,trunk,linear=nothing)

`DeepONet` is differential neural operator focused for solving physic-informed parametric ODEs.

DeepONet uses two neural networks, referred to as the "branch" and "trunk", to approximate
the solution of a differential equation. The branch network takes the spatial variables as
input and the trunk network takes the temporal variables as input. The final output is
the dot product of the outputs of the branch and trunk networks.

DeepONet is composed of two separate neural networks referred to as the "branch" and "trunk",
respectively. The branch net takes on input represents a function  evaluated at a collection
of fixed locations in some boundsand returns a features embedding. The trunk net takes the
continuous coordinates as inputs, and outputs a features embedding. The final output of the
DeepONet, the outputs of the branch and trunk networks are merged together via a dot product.

## Positional Arguments
*  `branch`: A branch neural network.
*  `trunk`: A trunk neural network.

## Keyword Arguments
* `linear`: A linear layer to apply to the output of the branch and trunk networks.

## Example

```julia
branch = Lux.Chain(
    Lux.Dense(1, 10, Lux.tanh_fast),
    Lux.Dense(10, 10, Lux.tanh_fast),
    Lux.Dense(10, 10))
trunk = Lux.Chain(
    Lux.Dense(1, 10, Lux.tanh_fast),
    Lux.Dense(10, 10, Lux.tanh_fast),
    Lux.Dense(10, 10, Lux.tanh_fast))
linear = Lux.Chain(Lux.Dense(10, 1))

deeponet = DeepONet(branch, trunk; linear= linear)

a = rand(1, 50, 40)
b = rand(1, 1, 40)
x = (branch = a, trunk = b)
θ, st = Lux.setup(Random.default_rng(), deeponet)
y, st = deeponet(x, θ, st)
```

## References
* Lu Lu, Pengzhan Jin, George Em Karniadakis "DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators"
* Sifan Wang "Learning the solution operator of parametric partial differential equations with physics-informed DeepOnets"
"""
struct DeepONet{L <: Union{Nothing, Lux.AbstractExplicitLayer }} <: NeuralOperator
    branch::Lux.AbstractExplicitLayer
    trunk::Lux.AbstractExplicitLayer
    linear::L
end

function DeepONet(branch, trunk; linear=nothing)
    DeepONet(branch, trunk, linear)
end

function Lux.setup(rng::AbstractRNG, l::DeepONet)
    branch, trunk, linear = l.branch, l.trunk, l.linear
    θ_branch, st_branch = Lux.setup(rng, branch)
    θ_trunk, st_trunk = Lux.setup(rng, trunk)
    θ = (branch = θ_branch, trunk = θ_trunk)
    st = (branch = st_branch, trunk = st_trunk)
    if linear !== nothing
        θ_liner, st_liner = Lux.setup(rng, linear)
        θ = (θ..., liner = θ_liner)
        st = (st..., liner = st_liner)
    end
    θ, st
end

Lux.initialstates(::AbstractRNG, ::DeepONet) = NamedTuple()

@inline function (f::DeepONet)(x::NamedTuple, θ, st::NamedTuple)
    x_branch, x_trunk = x.branch, x.trunk
    branch, trunk = f.branch, f.trunk
    st_branch, st_trunk = st.branch, st.trunk
    θ_branch, θ_trunk = θ.branch, θ.trunk
    out_b, st_b = branch(x_branch, θ_branch, st_branch)
    out_t, st_t = trunk(x_trunk, θ_trunk, st_trunk)
    if f.linear !== nothing
        linear = f.linear
        θ_liner, st_liner = θ.liner, st.liner
        # out = sum(out_b .* out_t, dims = 1)
        out_ = out_b .* out_t
        out, st_liner = linear(out_, θ_liner, st_liner)
        out = sum(out, dims = 1)
        return out, (branch = st_b, trunk = st_t, liner = st_liner)
    else
        out = sum(out_b .* out_t, dims = 1)
        return out, (branch = st_b, trunk = st_t)
    end
end
