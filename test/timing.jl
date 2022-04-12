using LinearAlgebra
using LinearAlgebra
using SkylineSolvers.Chol
using SkylineSolvers.Colsol
using SkylineSolvers.Ldlt
using SymRCM
using SparseArrays
using DataDrop
using Profile
using ProfileView

K = DataDrop.retrieve_matrix("K.h5")
@show size(K)
I, J, V = findnz(K)     

sky = Chol.SkylineMatrix(I, J, V, size(K, 1))
@show Chol.nnz(sky)
@time Chol.factorize!(sky)

sky = Colsol.SkylineMatrix(I, J, V, size(K, 1))
@show Colsol.nnz(sky)
@time Colsol.factorize!(sky)

sky = Ldlt.SkylineMatrix(I, J, V, size(K, 1))
@show Ldlt.nnz(sky)
@time Ldlt.factorize!(sky)

# sky = SkylineMatrix(I, J, V, size(K, 1))
# @profview SkylineSolvers.ldlt_factorize!(sky)
# b = rand(size(A, 1))
# x = ldlt_solve(sky, b)
# @test norm(A \ b - x) / norm(x) < 1e-6

true