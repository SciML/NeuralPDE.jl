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
    out_initW = Flux.glorot_uniform, out_initb = Flux.zeros32,
    train_input_layer=true, laplacian_eigenfunctions=false, 
    domains=nothing::Union{Nothing, Vector{<:ClosedInterval}}, 
    boundary_conditions=nothing::Union{Nothing, Vector{Tuple{Symbol, Symbol}}})

    chains = map(1:length(in_vec)) do i
        layers = map(1:num_layers) do j
            rMFNLayer(in_vec[i], hid_vec[i], out_vec[i], j == 1, j == num_layers; 
                in_bias=in_bias, hid_bias=hid_bias, out_bias=out_bias, 
                in_initW=in_initW, in_initb=in_initb, 
                hid_initW=hid_initW, hid_initb=hid_initb,
                out_initW=out_initW, out_initb=out_initb,
                train_input_layer=train_input_layer, laplacian_eigenfunctions=laplacian_eigenfunctions,
                domains=domains, boundary_conditions=boundary_conditions)
        end
        FastChain(layers...)
    end
    chains
end

struct AddedFastChains{CT <: Tuple, IND <: Tuple} <: DiffEqFlux.FastLayer
    chains::CT
    param_indices::IND

    function AddedFastChains(chains...)
        current_index = 1
        param_indices_vec_current = UnitRange{Int}[]
        for i in 1:length(chains)
            next_index = current_index + (DiffEqFlux.paramlength(chains[i]) - 1)
            push!(param_indices_vec_current, current_index:next_index)
            current_index = next_index + 1
        end

        tuple_chains = tuple(chains...)
        tuple_param_indices = tuple(param_indices_vec_current...)
        new{typeof(tuple_chains), typeof(tuple_param_indices)}(tuple_chains, tuple_param_indices)
    end
end

DiffEqFlux.paramlength(f::AddedFastChains) = f.param_indices[end].stop
DiffEqFlux.initial_params(f::AddedFastChains) = vcat((DiffEqFlux.initial_params.(f.chains))...)
function (f::AddedFastChains)(x, p) 
    sum(
        f.chains[i](x, @view p[f.param_indices[i]])
        for i in 1:length(f.chains)
    )
end

function add_vector_fast_chains(vector_fast_chains...)
    length_chains = length(vector_fast_chains[1])
    for vector_chain in vector_fast_chains[2:end]
        if length(vector_chain) != length_chains
            throw("All vectors must have the same length")
        end
    end

    chains = map(1:length_chains) do i
        AddedFastChains([vector_fast_chains[j][i] for j in 1:length(vector_fast_chains)]...)
    end
    chains
end

function VectorOfMLP(input_dims::Vector{Int}, hidden_dims::Int, num_hid_layers::Int, nonlinfunc, initialparamsfunc)
    fastchains = FastChain[]
    initialparams = []

    for indim in input_dims # make a fastchain for this output
        fastchain_array = []

        # first layer
        if num_hid_layers > 0
            push!(fastchain_array, FastDense(indim, hidden_dims, nonlinfunc; initW=initialparamsfunc))
        end

        # hidden-hidden layers
        for _ in 2:(num_hid_layers - 1)
            push!(fastchain_array, FastDense(hidden_dims, hidden_dims, nonlinfunc; initW=initialparamsfunc))
        end

        # final layer, always 1 dim
        push!(fastchain_array, FastDense(hidden_dims, 1, identity; initW=initialparamsfunc)) 
        fastchain = FastChain(fastchain_array...)
        initialparam = DiffEqFlux.initial_params(fastchain)

        push!(fastchains, fastchain)
        push!(initialparams, initialparam)
    end

    (fastchains, initialparams)
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

struct rMFNLayer{P, IN, HID, OUT, INP, LAST} <: DiffEqFlux.FastLayer
    out_dim::Int
    hid_dim::Int
    in_dim::Int
    initial_params::P
    output_layer::OUT
    hidden_layer::HID
    input_layer::IN
    out_param_last_index::Int
    hid_param_last_index::Int
    in_param_last_index::Int
    in_layer_init::INP
    last_layer::LAST

    function rMFNLayer(in_dim::Integer, hid_dim::Integer, out_dim::Integer, is_first_layer::Bool, is_last_layer::Bool;
        in_bias = true, hid_bias = true, out_bias = true, 
        in_initW = Flux.glorot_uniform,  in_initb = Flux.zeros32,
        hid_initW = Flux.glorot_uniform, hid_initb = Flux.zeros32,
        out_initW = Flux.glorot_uniform, out_initb = Flux.zeros32,
        train_input_layer=true, laplacian_eigenfunctions=false, 
        domains=nothing::Union{Nothing, Vector{<:ClosedInterval}}, 
        boundary_conditions=nothing::Union{Nothing, Vector{Tuple{Symbol, Symbol}}})

        if laplacian_eigenfunctions
            if isnothing(domains)
                throw("Need to pass in the domains for the laplacian eigenfunctions")
            end
            if isnothing(boundary_conditions)
                throw("Need to pass in the boundary_conditions for the laplacian eigenfunctions")
            end
            num_eigenvalues = hid_dim
            input_layer = LaplacianEigenfunctionInputLayer(num_eigenvalues, domains, boundary_conditions)
            inner_hid_dim = input_layer.num_hidden
            inner_train_input_layer = false
        else
            input_layer = FastDense(in_dim, hid_dim, sin; bias=in_bias, initW=in_initW, initb=in_initb)
            inner_train_input_layer = train_input_layer
            inner_hid_dim = hid_dim
        end
        hid_fast_dense = is_first_layer ? nothing : FastDense(inner_hid_dim, inner_hid_dim, identity; bias=hid_bias, initW=hid_initW, initb=hid_initb)
        out_fast_dense = FastDense(inner_hid_dim, out_dim, identity; bias=out_bias, initW=out_initW, initb=out_initb)

        if inner_train_input_layer
            in_param_last_index = DiffEqFlux.paramlength(input_layer)
        else
            in_param_last_index = 0
        end
        hid_param_last_index = in_param_last_index + DiffEqFlux.paramlength(hid_fast_dense)
        out_param_last_index = hid_param_last_index + DiffEqFlux.paramlength(out_fast_dense)
        if inner_train_input_layer
            initial_params = vcat(DiffEqFlux.initial_params.((input_layer, hid_fast_dense, out_fast_dense))...)
        else
            initial_params = vcat(DiffEqFlux.initial_params.((hid_fast_dense, out_fast_dense))...)
        end
        in_layer_init = DiffEqFlux.initial_params(input_layer)

        is_last_layer_val = Val(is_last_layer)

        new{typeof(initial_params), typeof(input_layer), typeof(hid_fast_dense), typeof(out_fast_dense), typeof(in_layer_init), typeof(is_last_layer_val)}(
            out_dim, inner_hid_dim, in_dim, initial_params, out_fast_dense, hid_fast_dense, input_layer,
            out_param_last_index, hid_param_last_index, in_param_last_index, in_layer_init, is_last_layer_val)
    end
end


function generate_1d_vector_laplacian_eigenfunction(input_dimension::Integer, modes::Vector{<:Integer}, domain::ClosedInterval, boundary_conditions::Tuple{Symbol, Symbol})
    # generate_laplacian_eigenfunction(domains, boundary_conditions)
    # Generate a Laplacian eigenfunction for the given domains and boundary conditions.
    left_type, right_type = boundary_conditions
    domain_length = domain.right - domain.left
    left_start = domain.left
    
    if left_type == :neumann && right_type == :neumann
        mode_vector = modes
        w_multiplier = π / domain_length
        full_W = reshape(mode_vector .* w_multiplier, (length(modes), 1))
        b_adder = -left_start
        return (x, p) -> cos.(full_W .* (@view x[input_dimension:input_dimension, :]) .+ b_adder)
    elseif left_type == :dirichlet && right_type == :dirichlet
        mode_vector = modes .+ 1
        w_multiplier = π / domain_length
        full_W = reshape(mode_vector .* w_multiplier, (length(modes), 1))
        b_adder = -left_start
        @show full_W
        @show size(full_W)
        @show b_adder
        return (x, p) -> sin.(full_W .* (@view x[input_dimension:input_dimension, :]) .+ b_adder)
    elseif left_type == :dirichlet && right_type == :neumann
        mode_vector = (2 .* modes) .+ 1
        w_multiplier = π / (2domain_length)
        full_W = reshape(mode_vector .* w_multiplier, (length(modes), 1))
        b_adder = -left_start
        return (x, p) -> sin.(full_W .* (@view x[input_dimension:input_dimension, :]) .+ b_adder)
    elseif left_type == :neumann && right_type == :dirichlet
        mode_vector = (2 .* modes) .+ 1
        w_multiplier = π / (2domain_length)
        full_W = reshape(mode_vector .* w_multiplier, (length(modes), 1))
        b_adder = -left_start
        return (x, p) -> cos.(full_W .* (@view x[input_dimension:input_dimension, :]) .+ b_adder)
    end
end

struct LaplacianEigenfunctionInputLayer{FUN <: Function}
    full_func::FUN
    num_hidden::Int

    function LaplacianEigenfunctionInputLayer(num_eigenvalues_per_dim::Integer, domains::Vector{<: ClosedInterval}, boundary_conditions::Vector{Tuple{Symbol, Symbol}})
        # Generate a Laplacian eigenfunction for the given domains and boundary conditions.
        modes = vcat([0], map(i->2^i, 0:num_eigenvalues_per_dim-2))
        input_dim = length(domains)
        functions = map(1:input_dim) do i
            f = generate_1d_vector_laplacian_eigenfunction(i, modes, domains[i], boundary_conditions[i])
        end
        full_func = (x, p) -> begin
            LE_evals_1D = map(functions) do f 
                f(x, p)
            end
            LE_first_modes = map(1:input_dim) do i
                LE_evals_1D[i][1:1, :]
            end
            LE_first_modes_premixed = map(1:input_dim) do i
                working_mat = ones(eltype(LE_first_modes[1]), (1, size(LE_first_modes[1], 2)))
                for j in 1:input_dim
                    if i != j
                        working_mat = working_mat .* @view LE_first_modes[j][1:1, :]
                    end
                end
                working_mat
            end
            LE_evals_mixed = map(1:input_dim) do i
                LE_evals_1D[i] .* LE_first_modes_premixed[i]
            end
            vcat(LE_evals_mixed...)
        end
        num_hidden = num_eigenvalues_per_dim * input_dim
        new{typeof(full_func)}(full_func, num_hidden)
    end
end

(f::LaplacianEigenfunctionInputLayer)(x, p) = f.full_func(x, p)

DiffEqFlux.paramlength(f::rMFNLayer) = f.out_param_last_index
DiffEqFlux.initial_params(f::rMFNLayer) = f.initial_params

# (in)  g_i(x) = sin(ω_i * x .+ ϕ_i) 
# (hid) z_i = g_i(x) .* (W_i * z_(i-1) .+ b_i) 
# (out) y_i = y_(i-1) .+ (W_out_i * z_i .+ b_out_i) 
(f::rMFNLayer)(x::Number, p::T) where {T <: AbstractArray} = f(DiffEqBase.parameterless_type(p)([x]), p)

function (f::rMFNLayer)(x::AbstractArray, p)  
    if f.hidden_layer isa FastDense # not first layer
        throw("First rMFN layer should have is_first_layer = true")
    else
        g_i = f.input_layer(x, @view p[1:f.in_param_last_index])
        z_i = g_i
        y_i = f.output_layer(z_i, @view p[f.hid_param_last_index + 1:f.out_param_last_index])
        if f.last_layer isa Val{true}
            y_i
        else
            (x, z_i, y_i)
        end
    end
end

function (f::rMFNLayer)(xzy::Tuple{<:AbstractArray, <:AbstractArray, <:AbstractArray}, p)  
    if f.hidden_layer isa FastDense # not first layer
        x = xzy[1]
        z_prev = xzy[2]
        y_prev = xzy[3]
        g_i = f.input_layer(x, @view p[1:f.in_param_last_index])
        z_i = g_i .* f.hidden_layer(z_prev, @view p[f.in_param_last_index + 1:f.hid_param_last_index])
        y_i = y_prev .+ f.output_layer(z_i, @view p[f.hid_param_last_index + 1:f.out_param_last_index])
        if f.last_layer isa Val{true}
            y_i
        else
            (x, z_i, y_i)
        end
    else
        throw("This rMFN layer should have is_first_layer = false")
    end
end



#in_fast_dense  = FastDense(2, 4, sin; bias=true, initW=Flux.glorot_uniform, initb=Flux.zeros32)
#lay = rMFNLayer(2, 4, 1, true, false)
#θ_0 = DiffEqFlux.initial_params(lay)
#x = [0.0, 1.0]
#x2 = [0.0 1.0 1.0; 1.0 0.0 1.0]
#lay(x, θ_0)

#in_vec = [2]
#hid_vec = [4]
#out_vec = [1]
#num_layers = 4
#chains = VectorOfrMFNChain(in_vec, hid_vec, out_vec, num_layers)
#θ2_0 = DiffEqFlux.initial_params(chains[1])
#chains[1](x, θ2_0)
#chains[1](x2, θ2_0)
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
