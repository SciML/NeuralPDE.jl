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

######################################################################################

#define the sampling function
function sampling_function(domain_interior, domain_boundary)

    # sampling from within domain
    
    t_interior = rand(Uniform(t_initial, t_term), domain_interior)
    S_interior = rand(Uniform(S_low, S_high*S_multiplier), domain_interior)

    # terminal sample
    t_boundary = t_term .*ones(domain_boundary)
    S_boundary = rand(Uniform(S_low, S_high*S_multiplier), domain_boundary)

    return t_interior', S_interior', t_boundary', S_boundary'
end

######################################################################################


#define the loss function

function loss_function(N)

    t_interior, S_interior, t_boundary, S_boundary = sampling_function(nSim_interior, nSim_terminal)

    # differential operator loss
    ϵ = 0.01
    model_output_interior = model(t_interior, S_interior)
    ∂g∂x = (model(t_interior, S_interior .+ ϵ) - model_output_interior)./ϵ
    ∂g∂t = (model(t_interior .+ ϵ, S_interior) - model_output_interior)./ϵ
    ∂g∂xx = (model(t_interior, S_interior .+ 2*ϵ) - 2*model(t_interior, S_interior .+ ϵ) + model_output_interior)./(ϵ^2)

    operator_loss_vec = ∂g∂t + r.*S_interior.*∂g∂x + (0.5*(sigma^2)).*(S_interior.^2).*∂g∂xx - r.*model_output_interior

    payoff = relu.(K .- S_interior)
    value = model(t_interior, S_interior)
    L1 = mean((operator_loss_vec.*(value-payoff)).^2)

    temp = relu.(operator_loss_vec)
    L2 = mean(temp.^2)

    V_ineq = relu.(-(payoff - value))
    L3 = mean(V_ineq.^2)

    target_payoff = relu.(K .- S_boundary)
    fitted_payoff = model(t_boundary, S_boundary)

    L4 = mean((fitted_payoff - target_payoff).^2)

    return L1 + L2 + L3 + L4
end

##########################################################################################

# Problem Definition - Initialization
r = 0.05           # Interest rate
sigma = 0.5       # Volatility
K = 50             # Strike
t_term = 1              # Terminal time
S0 = 50           # Initial price

# Solution parameters
t_initial = 0 + 1e-10    # time lower bound
S_low = 0.0 + 1e-10  # spot price lower bound
S_high = 2*K         # spot price upper bound

# Analytical Solution - European put
function european_put(S, K, r, sigma, t)

    d1 = (log.(S./K) .+ (r + sigma^2/2)*(t_term-t))/(sigma*sqrt(t_term-t))
    d2 = d1 .- (sigma*sqrt(t_term-t))
    put_price = -S.*cdf.(Normal(0,1), -d1) .+ K*exp(-r * (t_term-t))*cdf.(Normal(0,1), -d2)
 
    return put_price
 end

 ########################################################################################

 # Neural Network Definition

n_steps = 10000
num_layers = 3
nodes_per_layer = 50
learning_rate = 0.001

# Training parameters
sampling_stages  = 100   # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 1000
nSim_terminal = 100
S_multiplier  = 1.5   # multiplier for oversampling i.e. draw S from [S_low, S_high * S_multiplier]

# define model
model = DGM(nodes_per_layer, num_layers, 1)
