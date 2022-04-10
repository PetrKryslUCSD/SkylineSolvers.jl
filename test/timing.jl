using LinearAlgebra
using LinearAlgebra
using SkylineLDLT
using SkylineLDLT: SkylineMatrix, ldlt_factorize!, ldlt_solve
using SymRCM
using SparseArrays
using DataDrop
using Profile
using ProfileView

K = DataDrop.retrieve_matrix("K.h5")
I, J, V = findnz(K)     
sky = SkylineMatrix(I, J, V, size(K, 1))
@time SkylineLDLT.ldlt_factorize!(sky)
sky = SkylineMatrix(I, J, V, size(K, 1))
@profview SkylineLDLT.ldlt_factorize!(sky)
# b = rand(size(A, 1))
# x = ldlt_solve(sky, b)
# @test norm(A \ b - x) / norm(x) < 1e-6

true