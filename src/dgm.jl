@concrete struct DGMLSTMLayer <: AbstractLuxLayer
    activation1
    activation2
    in_dims::Int
    out_dims::Int
    init_weight
    init_bias
end

function DGMLSTMLayer(in_dims::Int, out_dims::Int, activation1, activation2;
        init_weight = glorot_uniform, init_bias = zeros32)
    return DGMLSTMLayer(activation1, activation2, in_dims, out_dims, init_weight, init_bias)
end

function initialparameters(rng::AbstractRNG, l::DGMLSTMLayer)
    return (;
        Uz = l.init_weight(rng, l.out_dims, l.in_dims),
        Ug = l.init_weight(rng, l.out_dims, l.in_dims),
        Ur = l.init_weight(rng, l.out_dims, l.in_dims),
        Uh = l.init_weight(rng, l.out_dims, l.in_dims),
        Wz = l.init_weight(rng, l.out_dims, l.out_dims),
        Wg = l.init_weight(rng, l.out_dims, l.out_dims),
        Wr = l.init_weight(rng, l.out_dims, l.out_dims),
        Wh = l.init_weight(rng, l.out_dims, l.out_dims),
        bz = l.init_bias(rng, l.out_dims),
        bg = l.init_bias(rng, l.out_dims),
        br = l.init_bias(rng, l.out_dims),
        bh = l.init_bias(rng, l.out_dims)
    )
end

function parameterlength(l::DGMLSTMLayer)
    return 4 * (l.out_dims * l.in_dims + l.out_dims * l.out_dims + l.out_dims)
end

# TODO: use more optimized versions from LuxLib
# XXX: Why not use the one from Lux?
function (layer::DGMLSTMLayer)((S, x), ps, st::NamedTuple)
    (; Uz, Ug, Ur, Uh, Wz, Wg, Wr, Wh, bz, bg, br, bh) = ps
    Z = layer.activation1.(Uz * x .+ Wz * S .+ bz)
    G = layer.activation1.(Ug * x .+ Wg * S .+ bg)
    R = layer.activation1.(Ur * x .+ Wr * S .+ br)
    H = layer.activation2.(Uh * x .+ Wh * (S .* R) .+ bh)
    S_new = (1 .- G) .* H .+ Z .* S
    return S_new, st
end

dgm_lstm_block_rearrange(Sᵢ₊₁, (Sᵢ, x)) = Sᵢ₊₁, x

function DGMLSTMBlock(layers...)
    blocks = AbstractLuxLayer[]
    for (i, layer) in enumerate(layers)
        if i == length(layers)
            push!(blocks, layer)
        else
            push!(blocks, SkipConnection(layer, dgm_lstm_block_rearrange))
        end
    end
    return Chain(blocks...)
end

@concrete struct DGM <: AbstractLuxWrapperLayer{:model}
    model
end

"""
    DGM(in_dims::Int, out_dims::Int, modes::Int, L::Int, activation1, activation2,
        out_activation=identity)

returns the architecture defined for Deep Galerkin method.

```math
\\begin{align}
S^1 &= \\sigma_1(W^1 x + b^1); \\
Z^l &= \\sigma_1(U^{z,l} x + W^{z,l} S^l + b^{z,l}); \\quad l = 1, \\ldots, L; \\
G^l &= \\sigma_1(U^{g,l} x + W^{g,l} S_l + b^{g,l}); \\quad l = 1, \\ldots, L; \\
R^l &= \\sigma_1(U^{r,l} x + W^{r,l} S^l + b^{r,l}); \\quad l = 1, \\ldots, L; \\
H^l &= \\sigma_2(U^{h,l} x + W^{h,l}(S^l \\cdot R^l) + b^{h,l}); \\quad l = 1, \\ldots, L; \\
S^{l+1} &= (1 - G^l) \\cdot H^l + Z^l \\cdot S^{l}; \\quad l = 1, \\ldots, L; \\
f(t, x, \\theta) &= \\sigma_{out}(W S^{L+1} + b).
\\end{align}
```

## Positional Arguments:

- `in_dims`: number of input dimensions = (spatial dimension + 1).
- `out_dims`: number of output dimensions.
- `modes`: Width of the LSTM type layer (output of the first Dense layer).
- `layers`: number of LSTM type layers.
- `activation1`: activation function used in LSTM type layers.
- `activation2`: activation function used for the output of LSTM type layers.
- `out_activation`: activation fn used for the output of the network.
- `kwargs`: additional arguments to be splatted into [`PhysicsInformedNN`](@ref).
"""
function DGM(in_dims::Int, out_dims::Int, modes::Int, layers::Int,
        activation1, activation2, out_activation)
    return DGM(Chain(
        SkipConnection(
            Dense(in_dims => modes, activation1),
            DGMLSTMBlock([DGMLSTMLayer(in_dims, modes, activation1, activation2)
                          for _ in 1:layers]...)),
        Dense(modes => out_dims, out_activation)))
end

"""
    DeepGalerkin(in_dims::Int, out_dims::Int, modes::Int, L::Int, activation1::Function,
        activation2::Function, out_activation::Function, strategy::AbstractTrainingStrategy;
        kwargs...)

## Arguments:

- `in_dims`: number of input dimensions = (spatial dimension + 1).
- `out_dims`: number of output dimensions.
- `modes`: Width of the LSTM type layer.
- `L`: number of LSTM type layers.
- `activation1`: activation fn used in LSTM type layers.
- `activation2`: activation fn used for the output of LSTM type layers.
- `out_activation`: activation fn used for the output of the network.
- `kwargs`: additional arguments to be splatted into [`PhysicsInformedNN`](@ref).

## Examples

```julia
discretization = DeepGalerkin(2, 1, 30, 3, tanh, tanh, identity, QuasiRandomTraining(4_000))
```
## References

Sirignano, Justin and Spiliopoulos, Konstantinos, "DGM: A deep learning algorithm for solving partial differential equations",
Journal of Computational Physics, Volume 375, 2018, Pages 1339-1364, doi: https://doi.org/10.1016/j.jcp.2018.08.029
"""
function DeepGalerkin(
        in_dims::Int, out_dims::Int, modes::Int, L::Int, activation1::Function,
        activation2::Function, out_activation::Function, strategy::AbstractTrainingStrategy;
        kwargs...)
    return PhysicsInformedNN(
        DGM(in_dims, out_dims, modes, L, activation1, activation2, out_activation),
        strategy; kwargs...
    )
end
