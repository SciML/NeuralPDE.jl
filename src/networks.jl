"""
Implementation of the Residual Multiplicative Filter Network from the paper:

Residual Multiplicative Filter Networks for Multiscale Reconstruction
Shayan Shekarforoush, David B. Lindell, David J. Fleet, Marcus A. Brubaker
https://arxiv.org/abs/2206.00746


"""
function VectorOfrMFNChain(in_vec, hid_vec, out_vec, num_layers; 
    in_bias = true, hid_bias = true, out_bias = true, 
    in_initW = Flux.glorot_uniform,  in_initb = Flux.zeros32,
    hid_initW = Flux.glorot_uniform, hid_initb = Flux.zeros32,
    out_initW = Flux.glorot_uniform, out_initb = Flux.zeros32)

    chains = map(1:length(in_vec)) do i
        layers = map(1:num_layers) do j
            rMFNLayer(in_vec[i], hid_vec[i], out_vec[i], j == 1, j == num_layers; 
                in_bias=in_bias, hid_bias=hid_bias, out_bias=out_bias, 
                in_initW=in_initW, in_initb=in_initb, 
                hid_initW=hid_initW, hid_initb=hid_initb,
                out_initW=out_initW, out_initb=out_initb)
        end
        FastChain(layers...)
    end
    chains
end

# params:
# x: R^d_in
# ω_i: R^(d_h x d_in)
# ϕ_i: R^(d_h)
# W_i: R^(d_h x d_h)
# b_i: R^(d_h)
# W_out_i: R^(d_out x d_h)
# b_out_i: R^(d_out)

# (in)  g_i(x) = sin(ω_i * x .+ ϕ_i) 
# (hid) z_i = g_i(x) .* (W_i * z_(i-1) .+ b_i) 
# (out) y_i = y_(i-1) .+ (W_out_i * z_i .+ b_out_i) 

struct rMFNLayer{P, IN, HID, OUT, INP} <: DiffEqFlux.FastLayer
    out::Int
    hid::Int
    in::Int
    initial_params::P
    out_fast_dense::OUT
    hid_fast_dense::HID
    in_fast_dense::IN
    out_param_last_index::Int
    hid_param_last_index::Int
    in_param_last_index::Int
    last_layer::Bool
    in_layer_init::INP

    function rMFNLayer(in::Integer, hid::Integer, out::Integer, first_layer::Bool, last_layer::Bool;
        in_bias = true, hid_bias = true, out_bias = true, 
        in_initW = Flux.glorot_uniform,  in_initb = Flux.zeros32,
        hid_initW = Flux.glorot_uniform, hid_initb = Flux.zeros32,
        out_initW = Flux.glorot_uniform, out_initb = Flux.zeros32,
        train_input_layer=true)

        in_fast_dense  = FastDense(in, hid, sin; bias=in_bias, initW=in_initW, initb=in_initb)
        hid_fast_dense = first_layer ? nothing : FastDense(hid, hid, identity; bias=hid_bias, initW=hid_initW, initb=hid_initb)
        out_fast_dense = FastDense(hid, out, identity; bias=out_bias, initW=out_initW, initb=out_initb)

        if train_input_layer
            in_param_last_index = DiffEqFlux.paramlength(in_fast_dense)
        else
            in_param_last_index = 0
        end
        hid_param_last_index = in_param_last_index + DiffEqFlux.paramlength(hid_fast_dense)
        out_param_last_index = hid_param_last_index + DiffEqFlux.paramlength(out_fast_dense)
        if train_input_layer
            initial_params = vcat(DiffEqFlux.initial_params.((in_fast_dense, hid_fast_dense, out_fast_dense))...)
        else
            initial_params = vcat(DiffEqFlux.initial_params.((hid_fast_dense, out_fast_dense))...)
        end
        in_layer_init = DiffEqFlux.initial_params(in_fast_dense)

        new{typeof(initial_params), typeof(in_fast_dense), typeof(hid_fast_dense), typeof(out_fast_dense), typeof(in_layer_init)}(
            out, hid, in, initial_params, out_fast_dense, hid_fast_dense, in_fast_dense,
            out_param_last_index, hid_param_last_index, in_param_last_index, last_layer, in_layer_init)
    end
end

function convert_boundary_conditions_to_Laplacian_Eigenfunction_initW_initb(domains::Vector{<: ClosedInterval}, boundary_conditions::Vector{Tuple{Symbol, Symbol}})
end

struct LaplacianEigenfunction{}
end

function RectangularLaplacianEigenfunctionrMFNLayer(in::Integer, num_eigenvalues_per_dim::Integer, out::Integer, 
        first_layer::Bool, last_layer::Bool, domains::Vector{<: ClosedInterval}, boundary_conditions::Vector{Tuple{Symbol, Symbol}};
        hid_bias = true, out_bias = true, 
        hid_initW = Flux.glorot_uniform, hid_initb = Flux.zeros32,
        out_initW = Flux.glorot_uniform, out_initb = Flux.zeros32)

        hid = num_eigenvalues_per_dim * in


        # need to make sure that we multiply the fastdenses for each input with a mode index 0 version
        # assumes shared w and b
        #map(1:in) do k
        #end
        W = zeros(Float32, num_eigenvalues_per_dim, in)
        b = zeros(Float32, num_eigenvalues_per_dim)

        for i in 1:num_eigenvalues_per_dim, j in 1:in, k in 1:in
            length = domains[j].right - domains[j].left
            mode_index = j == k ? 2^(i-1) : 1 
            if boundary_conditions[k][1] == :neumann && boundary_conditions[k][2] == :neumann
                mode_scale = (mode_index - 1)
                scale = 1
                phase = 1//2 * π
            elseif boundary_conditions[k][1] == :dirichlet && boundary_conditions[k][2] == :dirichlet
                mode_scale = mode_index
                scale = 1
                phase = 0
            elseif boundary_conditions[k][1] == :dirichlet && boundary_conditions[k][2] == :neumann
                mode_scale = (2mode_index - 1)
                scale = 1//2
                phase = 0
            elseif boundary_conditions[k][1] == :neumann && boundary_conditions[k][2] == :dirichlet
                mode_scale = (2mode_index - 1)
                scale = 1//2
                phase = 1//2 * π
            else
                error("unsupported boundary condition for domain $j, bc: $(boundary_conditions[j])")
            end
            if j == k
                W_unrolled[i, j, k] =  mode_scale * scale * π / length
            else
                W_unrolled[i, j, k] =  mode_scale * scale * π / length
            end
            b_unrolled[i, j] = -domains[j].left * mode_scale * scale * π / length + phase
        end
        W = reshape(W_unrolled, hid, in)
        b = reshape(b_unrolled, hid)
        """
        if left_type == :neumann && right_type == :neumann
            return x -> sin(m * π * (x - domains.left) / length + π / 2)
        elseif left_type == :dirichlet && right_type == :dirichlet
            return x -> sin((m + 1) * π * x / length)
        elseif left_type == :dirichlet && right_type == :neumann
            return x -> sin((2m + 1) * π / 2 * x / length)
        elseif left_type == :neumann && right_type == :dirichlet
            return x -> sin((2m + 1) * π / 2 * x / length + π / 2)
        end
        """
        initW(_, _) = copy(W)
        initb(_) = copy(b)
        

        return initW, initb

        
end


DiffEqFlux.paramlength(f::rMFNLayer) = f.out_param_last_index
DiffEqFlux.initial_params(f::rMFNLayer) = f.initial_params

# (in)  g_i(x) = sin(ω_i * x .+ ϕ_i) 
# (hid) z_i = g_i(x) .* (W_i * z_(i-1) .+ b_i) 
# (out) y_i = y_(i-1) .+ (W_out_i * z_i .+ b_out_i) 
(f::rMFNLayer)(x::Number, p::T) where {T <: AbstractArray} = f(DiffEqBase.parameterless_type(p)([x]), p)

function (f::rMFNLayer)(x::AbstractArray, p)  
    if f.hid_fast_dense isa FastDense # not first layer
        throw("First rMFN layer should have first_layer = true")
    else
        g_i = f.in_fast_dense(x, @view p[1:f.in_param_last_index])
        z_i = g_i
        y_i = f.out_fast_dense(z_i, @view p[f.hid_param_last_index + 1:f.out_param_last_index])
        if f.last_layer
            y_i
        else
            (x, z_i, y_i)
        end
    end
end

function (f::rMFNLayer)(xzy::Tuple{<:AbstractArray, <:AbstractArray, <:AbstractArray}, p)  
    if f.hid_fast_dense isa FastDense # not first layer
        x = xzy[1]
        z_prev = xzy[2]
        y_prev = xzy[3]
        g_i = f.in_fast_dense(x, @view p[1:f.in_param_last_index])
        z_i = g_i .* f.hid_fast_dense(z_prev, @view p[f.in_param_last_index + 1:f.hid_param_last_index])
        y_i = y_prev .+ f.out_fast_dense(z_i, @view p[f.hid_param_last_index + 1:f.out_param_last_index])
        if f.last_layer
            y_i
        else
            (x, z_i, y_i)
        end
    else
        throw("This rMFN layer should have first_layer = false")
    end
end

struct LErMFNLayer{P, IN, HID, OUT} <: DiffEqFlux.FastLayer
    inner_layer::rMFNLayer{P, IN, HID, OUT}
    num_exp_eigenvalues::Int

    function LErMFNLayer(in::Integer, num_eigenfunctions::Integer, out::Integer, first_layer::Bool, last_layer::Bool)
        #in::Integer, hid::Integer, out::Integer, first_layer::Bool, last_layer::Bool;
        #in_bias = true, hid_bias = true, out_bias = true, 
        #in_initW = Flux.glorot_uniform,  in_initb = Flux.zeros32,
        #hid_initW = Flux.glorot_uniform, hid_initb = Flux.zeros32,
        #out_initW = Flux.glorot_uniform, out_initb = Flux.zeros32)

        #hidden_dim = 
        inner_layer = rMFNLayer(1, num_eigenfunctions, 8, first_layer, last_layer)
        new{typeof(initial_params), typeof(in_fast_dense), typeof(hid_fast_dense), typeof(out_fast_dense)}(
            out, hid, in, initial_params, out_fast_dense, hid_fast_dense, in_fast_dense,
            out_param_last_index, hid_param_last_index, in_param_last_index, last_layer)
    end
end


in_fast_dense  = FastDense(2, 4, sin; bias=true, initW=Flux.glorot_uniform, initb=Flux.zeros32)
lay = rMFNLayer(2, 4, 1, true, false)
θ_0 = DiffEqFlux.initial_params(lay)
x = [0.0, 1.0]
x2 = [0.0 1.0 1.0; 1.0 0.0 1.0]
lay(x, θ_0)

in_vec = [2]
hid_vec = [4]
out_vec = [1]
num_layers = 4
chains = VectorOfrMFNChain(in_vec, hid_vec, out_vec, num_layers)
θ2_0 = DiffEqFlux.initial_params(chains[1])
chains[1](x, θ2_0)
chains[1](x2, θ2_0)
nothing


"""
other papers:

BACON: Band-limited Coordinate Networks for Multiscale Scene Representation
David B. Lindell, Dave Van Veen, Jeong Joon Park, Gordon Wetzstein
https://arxiv.org/abs/2112.04645

SAPE: Spatially-Adaptive Progressive Encoding for Neural Optimization
Amir Hertz, Or Perel, Raja Giryes, Olga Sorkine-Hornung, Daniel Cohen-Or
https://arxiv.org/abs/2104.09125

Multiplicative Filter Networks
Rizal_Fathony, Anit Kumar Sahu, Devin Willmott, J Zico Kolter
https://openreview.net/forum?id=OmtmcPkkhT

Implicit neural representations with periodic activation functions
Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, Gordon Wetzstein
https://arxiv.org/abs/2006.09661
"""
nothing
