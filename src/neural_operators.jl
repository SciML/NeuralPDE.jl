#TODO: Add docstrings
"""
DeepONet(branch,trunk)
"""
struct DeepONet{} <: Lux.AbstractExplicitLayer
    branch::Lux.AbstractExplicitLayer
    trunk::Lux.AbstractExplicitLayer
end

function Lux.setup(rng::AbstractRNG, l::DeepONet)
    branch, trunk = l.branch, l.trunk
    θ_branch, st_branch = Lux.setup(rng, branch)
    θ_trunk, st_trunk = Lux.setup(rng, trunk)
    θ = (branch = θ_branch, trunk = θ_trunk)
    st = (branch = st_branch, trunk = st_trunk)
    θ, st
end

# function Lux.initialparameters(rng::AbstractRNG, e::DeepONet)
#     code
# end

Lux.initialstates(::AbstractRNG, ::DeepONet) = NamedTuple()

"""
example:

branch = Lux.Chain(Lux.Dense(1, 32, Lux.σ), Lux.Dense(32, 1))
trunk = Lux.Chain(Lux.Dense(1, 32, Lux.σ), Lux.Dense(32, 1))
a = rand(1, 100, 10)
t = rand(1, 1, 10)
x = (branch = a, trunk = t)

deeponet = DeepONet(branch, trunk)
θ, st = Lux.setup(Random.default_rng(), deeponet)
y = deeponet(x, θ, st)
"""
@inline function (f::DeepONet)(x::NamedTuple, θ, st::NamedTuple)
    parameters, cord = x.branch, x.trunk
    branch, trunk = f.branch, f.trunk
    st_branch, st_trunk = st.branch, st.trunk
    θ_branch, θ_trunk = θ.branch, θ.trunk
    out_b, st_b = branch(parameters, θ_branch, st_branch)
    out_t, st_t = trunk(cord, θ_trunk, st_trunk)
    out = out_b' * out_t
    return out, (branch = st_b, trunk = st_t)
end
