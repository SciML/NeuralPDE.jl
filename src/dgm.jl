using Flux

mutable struct LSTMLayer
    Ug::Array{Float64}
    Uz::Array{Float64}
    Ur::Array{Float64}
    Uh::Array{Float64}
    Wg::Array{Float64}
    Wz::Array{Float64}
    Wr::Array{Float64}
    Wh::Array{Float64}
    bg::Array{Float64}
    bz::Array{Float64}
    br::Array{Float64}
    bh::Array{Float64}
    act1
    act2
end

function LSTMLayer(input_dim::Integer, output_dim::Integer, activation_1, activation_2)
    Ug = Flux.glorot_uniform(output_dim, input_dim)
    Uz = Flux.glorot_uniform(output_dim, input_dim)
    Ur = Flux.glorot_uniform(output_dim, input_dim)
    Uh = Flux.glorot_uniform(output_dim, input_dim)
    Wg = Flux.glorot_uniform(output_dim, output_dim)
    Wz = Flux.glorot_uniform(output_dim, output_dim)
    Wr = Flux.glorot_uniform(output_dim, output_dim)
    Wh = Flux.glorot_uniform(output_dim, output_dim)
    bg = zeros(output_dim)
    bz = zeros(output_dim)
    br = zeros(output_dim)
    bh = zeros(output_dim)

    LSTMLayer(Ug, Uz, Ur, Uh, Wg, Wz, Wr, Wh, bg, bz, br, bh, activation_1, activation_2)
end

function (a::LSTMLayer)(S, x)
    Ug, Uz, Ur, Uh, Wg, Wz, Wr, Wh, bg, bz, br, bh, act1, act2 = a.Ug, a.Uz, a.Ur, a.Uh, a.Wg, a.Wz, a.Wr, a.Wh, a.bg, a.bz, a.br, a.bh, a.act1, a.act2

    G = act1.(Ug * x + Wg * S .+ bg)
    Z = act1.(Uz * x + Wz * S .+ bz)
    R = act1.(Ur * x + Wr * S .+ br)
    H = act2.(Uh * x + Wh * (S .* R) .+ bh)
    S_new = (1 .- G) .* H + Z .* S

    return S_new
end



################################################################

using Flux

mutable struct Dense_start
    W1::Array{Float64}
    b1::Array{Float64}
    act
end

function Dense_start(input_dim::Integer, output_dim::Integer, act)
    W1 = Flux.glorot_uniform(output_dim, input_dim)
    b1 = zeros(output_dim)
    
    Dense_start(W1, b1, act)
end

function (a::Dense_start)(x)
    W1, b1, act = a.W1, a.b1, a.act
    
    S_1 = act.(W1 * x .+ b1)
    
    return S_1
end


###############################################################

using Flux

mutable struct Dense_end
    W::Array{Float64}
    b::Array{Float64}
end

function Dense_end(input_dim::Integer, output_dim::Integer)
    W = Flux.glorot_uniform(output_dim, input_dim)
    b = zeros(output_dim)
    
    Dense_end(W, b)
end

function (a::Dense_end)(x)
    W, b = a.W, a.b
    
    y = W * x .+ b
    
    return y
end


#############################################################################


struct DGM
    D1::Dense_start
    LSTMLayers::Vector{LSTMLayer}
    D2::Dense_end
end


function DGM(layer_width, n_layers, spatial_dim)
    DGM(
        Dense_start(spatial_dim + 1, layer_width, tanh),
        [LSTMLayer(spatial_dim + 1, layer_width, tanh, tanh) for i in 1:n_layers],
        Dense_end(layer_width, 1)
    )
end

function (a::DGM)(t, x)
    D1, LSTMLayers, D2 = a.D1, a.LSTMLayers, a.D2

    X = vcat(t, x)

    S = D1(X)

    for lstm_layer in LSTMLayers
        S = lstm_layer(S, X)
    end

    y = D2(S)

    return y
end


Flux.@functor Dense_start
Flux.@functor LSTMLayer
Flux.@functor Dense_end
Flux.@functor DGM

# check if Flux recognizes parameters
model_1 = DGM(50, 3, 1)
Flux.params(model_1)