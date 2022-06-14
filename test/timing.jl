using LinearAlgebra
using SkylineSolvers
using SymRCM
using SparseArrays
using LDLFactorizations
using Sparspak
using DataDrop
using Profile
# using ProfileView
using UnicodePlots
using MKL

# const get_num_threads = function() # anonymous so it will be serialized when called
#     blas = LinearAlgebra.BLAS.vendor()
#     # Wrap in a try to catch unsupported blas versions
#     try
#         if blas == :openblas
#             return ccall((:openblas_get_num_threads, Base.libblas_name), Cint, ())
#         elseif blas == :openblas64
#             return ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
#         elseif blas == :mkl
#             return ccall((:MKL_Get_Max_Num_Threads, Base.libblas_name), Cint, ())
#         end

#         # OSX BLAS looks at an environment variable
#         if Sys.isapple()
#             return tryparse(Cint, get(ENV, "VECLIB_MAXIMUM_THREADS", "1"))
#         end
#     catch
#     end

#     return nothing
# end
# @show get_num_threads() 

K = DataDrop.retrieve_matrix("K63070.h5")
# K = DataDrop.retrieve_matrix("K28782.h5")
@show size(K)
@show nnz(K)
I, J, V = findnz(K)     

display(spy(K, canvas = DotCanvas))

# Test the factorization of the system matrix
# -------------------------------------------------------
# @info "$(@__FILE__): LDLFactorizations"
# @time factors = LDLFactorizations.ldl(K)
# @show nnz(factors)

# @info "$(@__FILE__): SkylineMatrix Colsol"
# sky = SkylineSolvers.Colsol.SkylineMatrix(I, J, V, size(K, 1))
# @show SkylineSolvers.Colsol.nnz(sky)
# @time SkylineSolvers.Colsol.factorize!(sky)

# @info "$(@__FILE__): SkylineMatrix chol"
# sky = SkylineSolvers.Chol.SkylineMatrix(I, J, V, size(K, 1))
# @show SkylineSolvers.Chol.nnz(sky)
# @time SkylineSolvers.Chol.factorize!(sky)

# @info "$(@__FILE__): SkylineMatrix 4"
# sky = SkylineSolvers.Ldlt4.SkylineMatrix(I, J, V, size(K, 1))
# @show SkylineSolvers.Ldlt4.nnz(sky)
# @time SkylineSolvers.Ldlt4.factorize!(sky)

# @info "$(@__FILE__): SkylineMatrix 3"
# sky = SkylineSolvers.Ldlt3.SkylineMatrix(I, J, V, size(K, 1))
# @show SkylineSolvers.Ldlt3.nnz(sky)
# @time SkylineSolvers.Ldlt3.factorize!(sky)

@info "$(@__FILE__): SkylineMatrix"
sky = SkylineSolvers.Ldlt.SkylineMatrix(I, J, V, size(K, 1))
@show SkylineSolvers.Ldlt.nnz(sky)
@time SkylineSolvers.Ldlt.factorize!(sky)

# @info "$(@__FILE__): SkylineMatrix 2"
# sky = SkylineSolvers.Ldlt2.SkylineMatrix(I, J, V, size(K, 1))
# @show SkylineSolvers.Ldlt2.nnz(sky)
# @time SkylineSolvers.Ldlt2.factorize!(sky)

# @info "$(@__FILE__): Sparspak"
# p = Sparspak.SpkProblem.Problem(size(K)...)
# @time Sparspak.SpkProblem.insparse!(p, I, J, V);
# @time s = Sparspak.SpkSparseSolver.SparseSolver(p);
# @time Sparspak.SpkSparseSolver.findorder!(s)
# @time Sparspak.SpkSparseSolver.symbolicfactor!(s)
# @time Sparspak.SpkSparseSolver.inmatrix!(s, p)
# BLAS.set_num_threads(4)
# @time Sparspak.SpkSparseSolver.factor!(s)
# @time Sparspak.SpkSparseSolver.solve!(s, p);

@info "$(@__FILE__): lu"
@time f = lu(K);



# sky = SkylineSolvers.Ldlt3.SkylineMatrix(I, J, V, size(K, 1))
# @show SkylineSolvers.Ldlt3.nnz(sky)
# @time SkylineSolvers.Ldlt3.factorize!(sky)

# sky = SkylineSolvers.Chol.SkylineMatrix(I, J, V, size(K, 1))
# @show SkylineSolvers.Chol.nnz(sky)
# @time SkylineSolvers.Chol.factorize!(sky)

# sky = SkylineSolvers.Colsol.SkylineMatrix(I, J, V, size(K, 1))
# @show SkylineSolvers.Colsol.nnz(sky)
# @time SkylineSolvers.Colsol.factorize!(sky)

# sky = SkylineSolvers.Ldlt.SkylineMatrix(I, J, V, size(K, 1))
# @show SkylineSolvers.Ldlt.nnz(sky)
# @time SkylineSolvers.Ldlt.factorize!(sky)

# sky = SkylineSolvers.Ldlt2.SkylineMatrix(I, J, V, size(K, 1))
# @show SkylineSolvers.Ldlt2.nnz(sky)
# @time SkylineSolvers.Ldlt2.factorize!(sky)



# Test the solution of a system of linear equations
# -------------------------------------------------------

# sky =  SkylineSolvers.Ldlt2.SkylineMatrix(I, J, V, size(K, 1))
# SkylineSolvers.Ldlt2.factorize!(sky)
# b = rand(size(K, 1))
# x = SkylineSolvers.Ldlt2.solve(sky, b)
# @show norm(K \ b - x) / norm(x) < 1e-6

true