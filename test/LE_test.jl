using Plots

LaplacianEigenfunction1D(m::Integer, length, both_type) = LaplacianEigenfunction1D(m, length, both_type, both_type)
function LaplacianEigenfunction1D(m::Integer, length, left_type, right_type)
    if left_type == :neumann && right_type == :neumann
        return x -> cos(m * π * x / length)
    elseif left_type == :dirichlet && right_type == :dirichlet
        return x -> sin((m + 1) * π * x / length)
    elseif left_type == :dirichlet && right_type == :neumann
        return x -> sin((2m + 1) * π / 2 * x / length)
    elseif left_type == :neumann && right_type == :dirichlet
        return x -> cos((2m + 1) * π / 2 * x / length)
    end
end

a(m) = (m + 1) * π 
b(m) = (m + 1) * π
a(1) - b(1) + π/2
a(1) + b(1) - π/2

laplacian_eigenfunctions_nn = map(0:9) do i
    LaplacianEigenfunction1D(i, 1, :neumann, :neumann)
end

laplacian_eigenfunctions_dd = map(0:20) do i
    LaplacianEigenfunction1D(i, 1, :dirichlet, :dirichlet)
end

laplacian_eigenfunctions_dn = map(0:9) do i
    LaplacianEigenfunction1D(i, 1, :dirichlet, :neumann)
end

laplacian_eigenfunctions_nd = map(0:9) do i
    LaplacianEigenfunction1D(i, 1, :neumann, :dirichlet)
end

ts = range(0.0, 1.0, 1000)
nn_evals = [f.(ts) for f in laplacian_eigenfunctions_nn]
nn_fullmult = reduce((x, y)-> x.*y, nn_evals)
plot(ts, nn_fullmult)
dd_evals = [f.(ts) for f in laplacian_eigenfunctions_dd]
dd_fullmult = reduce((x, y)-> x.*y, dd_evals)
plot(ts, dd_fullmult)
dn_evals = [f.(ts) for f in laplacian_eigenfunctions_dn]
nd_evals = [f.(ts) for f in laplacian_eigenfunctions_nd]
dd_com = dd_evals[1] .* dd_evals[1] .* dd_evals[1] .* dd_evals[1] .* dd_evals[1]
plot(ts, dd_com)
neumann_evalscombo = neumann_evals[2] .* neumann_evals[6]
plot(ts, dirichlet_evalscombo; legend=false)
plot(ts, nn_evals; legend=false)
plot(ts, dirichlet_evals; legend=false)
plot(ts, neumann_evalscombo; legend=false)
plot(ts, dn_evals; legend=false)
plot(ts, nd_evals; legend=false)
dn_com = dn_evals[4] .* dn_evals[1] .* dn_evals[1] .* dn_evals[1] .* dn_evals[1]
plot(ts, dn_com; legend=false)
plot(ts, dn_evals[1]; legend=false)

dd_com = dd_evals[3] .* dd_evals[6] #.* dd_evals[2]
dd_com = dd_evals[3] 
dd_com = dd_evals[1] .* dd_evals[1] .* dd_evals[1] .* dd_evals[1]
dd_com_sep = dd_evals[3] .* dd_evals[6] .- 0.5
plot(ts, [dd_com, dd_evals[6], dd_evals[3], dd_evals[10], dd_com .- 0.5 .* dd_evals[10]])
plot(ts, [dd_evals[1], 0.5 .* dd_evals[2], dd_com])
i = 2
j = 2
nn_com = nn_evals[i] .* nn_evals[j] .* nn_evals[i] .* nn_evals[j]
nn_com_sub = nn_com .- 0.5 .* nn_evals[i+j-1]
plot(ts, [nn_evals[i], nn_evals[j], nn_com, nn_com_sub])
nothing


dn(x, m) = sin((2m + 1) * x * π/2)
nd(x, m) = cos((2m + 1) * x * π/2)
plot(ts, dn.(ts, 3); legend=false)
plot(ts, nd.(ts, 1); legend=false)
nothing

