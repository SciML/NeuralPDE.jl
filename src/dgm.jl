mutable struct dgm_lstm_layer{T}
    Uz::AbstractMatrix{T}
    Ug::AbstractMatrix{T}
    Ur::AbstractMatrix{T}
    Uh::AbstractMatrix{T}
    Wz::AbstractMatrix{T}
    Wg::AbstractMatrix{T}
    Wr::AbstractMatrix{T}
    Wh::AbstractMatrix{T}
    bz::AbstractVector{T}
    bg::AbstractVector{T}
    br::AbstractVector{T}
    bh::AbstractVector{T}
    activation1::Function
    activation2::Function
end

function dgm_lstm_layer(input_dim :: Int, output_dim :: Int, activation_1, activation_2, dtype= Float64)
    dgm_lstm_layer(
        Flux.glorot_uniform(output_dim, input_dim), 
        Flux.glorot_uniform(output_dim, input_dim),
        Flux.glorot_uniform(output_dim, input_dim), 
        Flux.glorot_uniform(output_dim, input_dim), 
        Flux.glorot_uniform(output_dim, output_dim), 
        Flux.glorot_uniform(output_dim, output_dim),
        Flux.glorot_uniform(output_dim, output_dim), 
        Flux.glorot_uniform(output_dim, output_dim), 
        Flux.zeros32(output_dim), 
        Flux.zeros32(output_dim), 
        Flux.zeros32(output_dim), 
        Flux.zeros32(output_dim),
        activation_1, 
        activation_2)
end

function (a :: dgm_lstm_layer)(S::AbstractVecOrMat, x::AbstractVecOrMat)
    @unpack Ug, Uz, Ur, Uh, 
    Wg, Wz, Wr, Wh, 
    bg, bz, br, bh, 
    activation1, activation2 = a;

    G = activation1.(Ug*x .+ Wg*S .+ bg)
    Z = activation1.(Uz*x .+ Wz*S .+ bz)
    R = activation1.(Ur*x .+ Wr*S .+ br)
    H = activation2.(Uh*x .+ Wh*(S.*R) .+ bh)
    S_new = (1 .- G).*H .+ Z.*S
    return S_new
end

mutable struct dgm{L1, L2, L3}
    dense_start :: L1
    lstm_Layers :: L2
    dense_end :: L3
end

function dgm(in_dims::Int, out_dims::Int, modes::Int, L::Int, activation1, activation2, out_activation)
    dgm(
        Flux.Dense(in_dims, modes, activation1),
        [dgm_lstm_layer(in_dims, modes, activation1, activation2) for i in 1:L],
        Flux.Dense(modes, out_dims, out_activation)
    )
end


function (l :: dgm)(x)
    S = l.dense_start(x)
    for lstm_layer in l.lstm_Layers
        S = lstm_layer(S, x)
    end
    y = l.dense_end(S)

    return y
end

Flux.@functor dgm_lstm_layer
Flux.@functor dgm

function DGM(in_dims::Int, out_dims::Int, modes::Int, L::Int, activation1::Function, activation2::Function, out_activation::Function, strategy::NeuralPDE.AbstractTrainingStrategy; kwargs...)
    PhysicsInformedNN(Flux.Chain(
        dgm(in_dims, out_dims, modes, L, activation1, activation2, out_activation)),
        strategy; kwargs...
    )
end
