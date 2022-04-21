
module mcolsol3005a
using Test
using LinearAlgebra
using SkylineSolvers.Colsol: SkylineMatrix, factorize!, solve
using SparseArrays
function test()
    A = [        
    5.0 -4.0  1.0 0.0
    -4.0 6.0 -4.0 1.0
    1.0 -4.0 6.0 -4.0
    0.0 1.0 -4.0 5.0
    ]
    D =  [5.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00
    0.00000e+00  2.80000e+00  0.00000e+00  0.00000e+00
    0.00000e+00  0.00000e+00  2.14286e+00  0.00000e+00
    0.00000e+00  0.00000e+00  0.00000e+00  8.33333e-01]

    Lt = [1.00000e+00  -8.00000e-01   2.00000e-01   0.00000e+00
    0.00000e+00   1.00000e+00  -1.14286e+00   3.57143e-01
    0.00000e+00   0.00000e+00   1.00000e+00  -1.33333e+00
    0.00000e+00   0.00000e+00   0.00000e+00   1.00000e+00  ]
    A = sparse(A)
    I, J, V = findnz(A)     
    sky = SkylineMatrix(I, J, V, size(A, 1))
    factorize!(sky)
    F = Matrix(sparse(sky; symm = false))
    D = tril(triu(F, 0), 0)
    Lt = triu(F, 1) + LinearAlgebra.I
    @test norm(Lt' * D * Lt -  A) < 1e-6 * norm(A)
    true
end
end
using .mcolsol3005a
mcolsol3005a.test()

module mcolsol001
using Test
using SkylineSolvers.Colsol: _update_skyline!

function test()
    dofnums = [3 4; 7 8; 11 12; 9 10; 5 6; 1 2]
    bars = [1 2; 1 6; 6 5; 5 2; 6 2; 2 3; 3 4; 2 4]
    mht = fill(0, maximum(dofnums[:]))
    for b in 1:size(bars, 1)
        _update_skyline!(mht, [d for d in dofnums[bars[b, :], :]])
    end
    # @show mht
    @test mht == [0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5]
    true
end
end
using .mcolsol001
mcolsol001.test()

module mcolsol003
using Test
using SkylineSolvers.Colsol: _update_skyline!, _diagonal_addresses
function test()
    # Example From Section 12.2 of Bathe Finite Element Procedures (1997)
    mht = [0, 1, 1, 3, 1, 3, 1, 3]
    maxa = _diagonal_addresses(mht)
    @test maxa == [1, 2, 4, 6, 10, 12, 16, 18, 22]
    true
end
end
using .mcolsol003
mcolsol003.test()

module mcolsol002
using Test
using SkylineSolvers.Colsol: _update_skyline!, _diagonal_addresses
function test()
    dofnums = [3 4; 7 8; 11 12; 9 10; 5 6; 1 2]
    bars = [1 2; 1 6; 6 5; 5 2; 6 2; 2 3; 3 4; 2 4]
    mht = fill(0, maximum(dofnums[:]))
    for b in 1:size(bars, 1)
        _update_skyline!(mht, [d for d in dofnums[bars[b, :], :]])
    end
    # @show mht
    @test mht == [0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5]
    maxa = _diagonal_addresses(mht)
    @test maxa == [1, 2, 4, 7, 11, 16, 22, 29, 37, 40, 44, 49, 55] 
    true
end
end
using .mcolsol002
mcolsol002.test()

module mcolsol004
using Test
using SkylineSolvers.Colsol: SkylineMatrix, findnz
using SparseArrays
function test()
    for s in [0.5, 0.2, 0.1, 0.05]
        for M in 17:13:1177
            A = sprand(M, M, s)
            A = A + A'
            I, J, V = findnz(A)     
            sky = SkylineMatrix(I, J, V, M)
            I, J, V = findnz(sky)
            B = sparse(I, J, V, M, M)
            @test A == B
        end
    end
    true
end
end
using .mcolsol004
mcolsol004.test()

module mcolsol004a
using Test
using SkylineSolvers.Colsol: SkylineMatrix, sparse
using SparseArrays
function test()
    for s in [0.5, 0.2, 0.1, 0.05]
        for M in 17:13:1177
            A = sprand(M, M, s)
            A = A + A'
            I, J, V = findnz(A)     
            sky = SkylineMatrix(I, J, V, M)
            B = sparse(sky)
            @test A == B
        end
    end
    true
end
end
using .mcolsol004a
mcolsol004a.test()

module mcolsol005
using Test
using LinearAlgebra
using SkylineSolvers.Colsol: SkylineMatrix, factorize!, solve
using SparseArrays
function test()
    A = [   
      5.0 -4.0  1.0 0.0
     -4.0 6.0 -4.0 1.0
     1.0 -4.0 6.0 -4.0
     0.0 1.0 -4.0 5.0
    ]
    A = sparse(A)
    I, J, V = findnz(A)     
    sky = SkylineMatrix(I, J, V, size(A, 1))
    factorize!(sky)
    F = Matrix(sparse(sky; symm = false))
     D = tril(triu(F, 0), 0)
     Lt = triu(F, 1) + LinearAlgebra.I
     @test norm(Lt' * D * Lt -  A) < 1e-6 * norm(A)
    true
end
end
using .mcolsol005
mcolsol005.test()

module mcolsol005a
using Test
using LinearAlgebra
using SkylineSolvers.Colsol: SkylineMatrix, factorize!, solve
using SparseArrays
function test()
    A = [
        2.0 -2.0 0.0 0.0 -1.0
        -2.0 3.0 -2.0 0.0 0.0
        0.0 -2.0 5.0 -3.0 0.0
        0.0 0.0 -3.0 10.0 4.0
        -1.0 0.0 0.0 4.0 10.0
        ]
    A = sparse(A)
    I, J, V = findnz(A)     
    sky = SkylineMatrix(I, J, V, size(A, 1))
    factorize!(sky)
    F = Matrix(sparse(sky; symm = false))
     D = tril(triu(F, 0), 0)
     Lt = triu(F, 1) + LinearAlgebra.I
     @test norm(Lt' * D * Lt -  A) < 1e-6 * norm(A)
    true
end
end
using .mcolsol005a
mcolsol005a.test()

module mcolsol005b
using Test
using LinearAlgebra
using SkylineSolvers.Colsol: SkylineMatrix, factorize!, solve
using SparseArrays
function test()
    A = [
        2.0 -2.0 0.0 0.0 -1.0
        -2.0 3.0 -2.0 0.0 0.0
        0.0 -2.0 5.0 -3.0 0.0
        0.0 0.0 -3.0 10.0 4.0
        -1.0 0.0 0.0 4.0 10.0
        ]
    A = sparse(A)
    I, J, V = findnz(A)     
    sky = SkylineMatrix(I, J, V, size(A, 1))
    factorize!(sky)
    F = Matrix(sparse(sky; symm = false))
     D = tril(triu(F, 0), 0)
     Lt = triu(F, 1) + LinearAlgebra.I
     @test norm(Lt' * D * Lt -  A) < 1e-6 * norm(A)
     b = rand(size(A, 1))
    x = solve(sky, b)
@test norm(A \ b - x) / norm(x) < 1e-6
    true
end
end
using .mcolsol005b
mcolsol005b.test()


module mcolsol005c
using Test
using LinearAlgebra
using SkylineSolvers.Colsol: SkylineMatrix, factorize!, solve
using SparseArrays
function test()
    for s in [0.5, 0.2, 0.1, 0.05]
        for M in 17:3:277
            A = sprand(M, M, s)
            A = A + A' + LinearAlgebra.I
            I, J, V = findnz(A)     
            sky = SkylineMatrix(I, J, V, M)
            factorize!(sky)
            b = rand(size(A, 1))
            x = solve(sky, b)
            @test norm(A \ b - x) / norm(x) < 1e-6
        end
    end
    true
end
end
using .mcolsol005c
mcolsol005c.test()

