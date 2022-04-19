using LinearAlgebra
using LinearAlgebra
using SkylineSolvers
using SymRCM
using SparseArrays
using LDLFactorizations
using DataDrop
using Profile
using ProfileView

K = DataDrop.retrieve_matrix("K63070.h5")
# K = DataDrop.retrieve_matrix("K28782.h5")
@show size(K)
I, J, V = findnz(K)     

@time LDLFactorizations.ldl(K)

sky = SkylineSolvers.Ldlt4.SkylineMatrix(I, J, V, size(K, 1))
@show SkylineSolvers.Ldlt4.nnz(sky)
@time SkylineSolvers.Ldlt4.factorize!(sky)

sky = SkylineSolvers.Ldlt3.SkylineMatrix(I, J, V, size(K, 1))
@show SkylineSolvers.Ldlt3.nnz(sky)
@time SkylineSolvers.Ldlt3.factorize!(sky)

sky = SkylineSolvers.Chol.SkylineMatrix(I, J, V, size(K, 1))
@show SkylineSolvers.Chol.nnz(sky)
@time SkylineSolvers.Chol.factorize!(sky)

sky = SkylineSolvers.Colsol.SkylineMatrix(I, J, V, size(K, 1))
@show SkylineSolvers.Colsol.nnz(sky)
@time SkylineSolvers.Colsol.factorize!(sky)

sky = SkylineSolvers.Ldlt.SkylineMatrix(I, J, V, size(K, 1))
@show SkylineSolvers.Ldlt.nnz(sky)
@time SkylineSolvers.Ldlt.factorize!(sky)

sky = SkylineSolvers.Ldlt2.SkylineMatrix(I, J, V, size(K, 1))
@show SkylineSolvers.Ldlt2.nnz(sky)
@time SkylineSolvers.Ldlt2.factorize!(sky)

# sky =  SkylineSolvers.Ldlt2.SkylineMatrix(I, J, V, size(K, 1))
# SkylineSolvers.Ldlt2.factorize!(sky)
# b = rand(size(K, 1))
# x = SkylineSolvers.Ldlt2.solve(sky, b)
# @show norm(K \ b - x) / norm(x) < 1e-6

true