struct dgm_lstm_layer{F1, F2} <:Lux.AbstractExplicitLayer
    activation1
    activation2
    in_dims::Int
    out_dims::Int
    init_weight::F1
    init_bias::F2
end


function dgm_lstm_layer(in_dims::Int, out_dims::Int, activation1, activation2;
    init_weight= Lux.glorot_uniform, init_bias= Lux.zeros32)
    return dgm_lstm_layer{typeof(init_weight), typeof(init_bias)}(activation1, activation2, in_dims, out_dims, init_weight, init_bias);
end

import Lux:initialparameters, initialstates, parameterlength, statelength

function Lux.initialparameters(rng::AbstractRNG, l::dgm_lstm_layer)
    return (
        Uz= l.init_weight(rng, l.out_dims, l.in_dims),
        Ug= l.init_weight(rng, l.out_dims, l.in_dims),
        Ur= l.init_weight(rng, l.out_dims, l.in_dims),
        Uh= l.init_weight(rng, l.out_dims, l.in_dims),
        Wz= l.init_weight(rng, l.out_dims, l.out_dims),
        Wg= l.init_weight(rng, l.out_dims, l.out_dims),
        Wr= l.init_weight(rng, l.out_dims, l.out_dims),
        Wh= l.init_weight(rng, l.out_dims, l.out_dims),
        bz= l.init_bias(rng, l.out_dims) ,
        bg= l.init_bias(rng, l.out_dims) ,
        br= l.init_bias(rng, l.out_dims) ,
        bh= l.init_bias(rng, l.out_dims) 
    )
end

Lux.initialstates(::AbstractRNG, ::dgm_lstm_layer)= NamedTuple()
Lux.parameterlength(l::dgm_lstm_layer)= 4* (l.out_dims* l.in_dims + l.out_dims* l.out_dims+ l.out_dims)
Lux.statelength(l::dgm_lstm_layer)= 0

function (layer::dgm_lstm_layer)(S::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, ps, st::NamedTuple) where T
    @unpack Uz, Ug, Ur, Uh, Wz, Wg, Wr, Wh, bz, bg, br, bh= ps
    Z= layer.activation1.(Uz*x+ Wz*S .+ bz);
    G= layer.activation1.(Ug*x+ Wg*S .+ bg);
    R= layer.activation1.(Ur*x+ Wr*S .+ br);
    H= layer.activation2.(Uh*x+ Wh*(S.*R) .+ bh);
    S_new= (1. .-G).*H .+ Z.*S;
    return S_new, st;
end

struct dgm_lstm_block{L <:NamedTuple} <: Lux.AbstractExplicitContainerLayer{(:layers,)} 
    layers::L
end

function dgm_lstm_block(l...)
    names= ntuple(i-> Symbol("dgm_lstm_$i"), length(l));
    layers= NamedTuple{names}(l);
    return dgm_lstm_block(layers);
end

dgm_lstm_block(xs::AbstractVector)=  dgm_lstm_block(xs...)

@generated function apply_dgm_lstm_block(layers::NamedTuple{fields}, S::AbstractVecOrMat, x::AbstractVecOrMat, ps, st::NamedTuple) where fields
    N= length(fields);
    S_symbols = vcat([:S], [gensym() for _ in 1:N])
    x_symbol= :x;
    st_symbols = [gensym() for _ in 1:N]
    calls = [:(($(S_symbols[i + 1]), $(st_symbols[i])) = layers.$(fields[i])(
        $(S_symbols[i]), $(x_symbol), ps.$(fields[i]), st.$(fields[i]))) for i in 1:N]
    push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
    push!(calls, :(return $(S_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

function (L::dgm_lstm_block)(S::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, ps, st::NamedTuple) where T
    return apply_dgm_lstm_block(L.layers, S, x, ps, st)
end

struct dgm{S, L, E} <: Lux.AbstractExplicitContainerLayer{(:d_start, :lstm, :d_end)}
    d_start::S
    lstm:: L
    d_end:: E
end

function (l::dgm)(x::AbstractVecOrMat{T}, ps, st::NamedTuple) where T

    S, st_start= l.d_start(x, ps.d_start, st.d_start);
    S, st_lstm= l.lstm(S, x, ps.lstm, st.lstm);
    y, st_end= l.d_end(S, ps.d_end, st.d_end);
    
    st_new= (
        d_start= st_start,
        lstm= st_lstm,
        d_end= st_end
    )
    return y, st_new;

end 
"""
`dgm(in_dims::Int, out_dims::Int, modes::Int, L::Int, activation1, activation2, out_activation= Lux.identity)`:
returns the architecture defined in https://arxiv.org/abs/1708.07469

### Arguments:

`in_dims`: number of input dimensions= (spatial dimension+ 1)
`out_dims`: number of output dimensions
`modes`: Width of the LSTM type layer
`L`: number of LSTM type layers
`activation1`: activation fn used in LSTM type layers
`activation2`: activation fn used for the output of LSTM type layers
`out_activation`: activation fn used for the output of the network

"""
function dgm(in_dims::Int, out_dims::Int, modes::Int, L::Int, activation1, activation2, out_activation)
    dgm(
        Lux.Dense(in_dims, modes, activation1),
        dgm_lstm_block([dgm_lstm_layer(in_dims, modes, activation1, activation2) for i in 1:L]),
        Lux.Dense(modes, out_dims, out_activation)
    )
end

"""
`DeepGalerkin(in_dims::Int, out_dims::Int, modes::Int, L::Int, activation1::Function, activation2::Function, out_activation::Function, 
    strategy::NeuralPDE.AbstractTrainingStrategy; kwargs...)`:
returns a `discretize` algorithm for the ModelingToolkit PDESystem interface, which transforms a `PDESystem` into an
    `OptimizationProblem` using the Deep Galerkin method.
### Arguments:

`in_dims`: number of input dimensions= (spatial dimension+ 1)
`out_dims`: number of output dimensions
`modes`: Width of the LSTM type layer
`L`: number of LSTM type layers
`activation1`: activation fn used in LSTM type layers
`activation2`: activation fn used for the output of LSTM type layers
`out_activation`: activation fn used for the output of the network
"""
function DeepGalerkin(in_dims::Int, out_dims::Int, modes::Int, L::Int, activation1::Function, activation2::Function, out_activation::Function, strategy::NeuralPDE.AbstractTrainingStrategy; kwargs...)
    PhysicsInformedNN(
        dgm(in_dims, out_dims, modes, L, activation1, activation2, out_activation),
        strategy; kwargs...
    )
end