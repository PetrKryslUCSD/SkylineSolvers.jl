module mchol001
using Test
using SkylineSolvers.Chol.Chol: update_skyline!

function test()
    dofnums = [3 4; 7 8; 11 12; 9 10; 5 6; 1 2]
    bars = [1 2; 1 6; 6 5; 5 2; 6 2; 2 3; 3 4; 2 4]
    skylngs = fill(0, maximum(dofnums[:]))
    for b in 1:size(bars, 1)
        update_skyline!(skylngs, [d for d in dofnums[bars[b, :], :]])
    end
    # @show skylngs
    @test skylngs == [1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6]
    true
end
end
using .mchol001
mchol001.test()

module mchol002
using Test
using SkylineSolvers.Chol: update_skyline!, diagonal_addresses

function test()
    dofnums = [3 4; 7 8; 11 12; 9 10; 5 6; 1 2]
    bars = [1 2; 1 6; 6 5; 5 2; 6 2; 2 3; 3 4; 2 4]
    skylngs = fill(0, maximum(dofnums[:]))
    for b in 1:size(bars, 1)
        update_skyline!(skylngs, [d for d in dofnums[bars[b, :], :]])
    end
    # @show skylngs
    @test skylngs == [1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6]
    d = diagonal_addresses(skylngs)
    @test d == [1, 3, 6, 10, 15, 21, 28, 36, 39, 43, 48, 54]  
    true
end
end
using .mchol002
mchol002.test()

module mchol004
using Test
using SkylineSolvers.Chol: SkylineMatrix, findnz
using SparseArrays
function test()
    for M in 17:13:177
        A = sprand(M, M, 0.1)
        A = A + A'
        I, J, V = findnz(A)     
        sky = SkylineMatrix(I, J, V, M)
        I, J, V = findnz(sky)
        B = sparse(I, J, V, M, M)
        @test A == B
    end
    true
end
end
using .mchol004
mchol004.test()

module mchol005
using Test
using LinearAlgebra
using SkylineSolvers.Chol: SkylineMatrix, factorize!, solve
using SparseArrays
function test()
    M = 8
        A = sparse([1, 4, 2, 6, 3, 4, 1, 3, 4, 7, 8, 5, 2, 6, 4, 7, 8, 4, 7, 8], [1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 8], [2.5, -8.34527e-02, 2.50000e+00, -3.85263e-02, 2.50000e+00, -8.31917e-02, -8.34527e-02, -8.31917e-02, 2.50000e+00, -6.75608e-02, -5.03247e-02, 2.50000e+00, -3.85263e-02, 2.50000e+00, -6.75608e-02, 2.50000e+00, -2.16746e-02, -5.03247e-02, -2.16746e-02, 2.47881e+00], 8, 8)    
        Matrix(A) = [
        2.5 
        0          2.5 
        0          0          2.5 
        -8.345e-02 0          -8.319e-02  2.5
        0          0          0           0            2.5
        0          -3.852e-02 0           0            0           2.5
        0          0          0          -6.756e-02   0           0           2.5 
        0          0          0          -5.032e-02   0           0          -2.167e-02 2.478e+00] 
     
        I, J, V = findnz(A)     
        sky = SkylineMatrix(I, J, V, M)
        factorize!(sky)
        
    true
end
end
using .mchol005
mchol005.test()


module mchol006
using Test
using LinearAlgebra
using SkylineSolvers.Chol: SkylineMatrix, factorize!, solve
using SparseArrays
function test()
    for M in 17:13:177
        A = sprand(M, M, 0.1)
        A = -0.1*(A + A') + 2.5*LinearAlgebra.I
        b = rand(M)
        I, J, V = findnz(A)     
        sky = SkylineMatrix(I, J, V, M)
        factorize!(sky)
        x = solve(sky, b)
        xt = A\b
        @test norm(x - xt) < 1e-6 * norm(xt)
    end
    true
end
end
using .mchol006
mchol006.test()
