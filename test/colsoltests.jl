module mbas001
using Test
using SkylineLDLT: update_skyline!

function test()
    dofnums = [3 4; 7 8; 11 12; 9 10; 5 6; 1 2]
    bars = [1 2; 1 6; 6 5; 5 2; 6 2; 2 3; 3 4; 2 4]
    mht = fill(0, maximum(dofnums[:]))
    for b in 1:size(bars, 1)
        update_skyline!(mht, [d for d in dofnums[bars[b, :], :]])
    end
    # @show mht
    @test mht == [0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5]
    true
end
end
using .mbas001
mbas001.test()

module mbas003
using Test
using SkylineLDLT: update_skyline!, diagonal_addresses
function test()
    # Example From Section 12.2 of Bathe Finite Element Procedures (1997)
    mht = [0, 1, 1, 3, 1, 3, 1, 3]
    maxa = diagonal_addresses(mht)
    @test maxa == [1, 2, 4, 6, 10, 12, 16, 18, 22]
    true
end
end
using .mbas003
mbas003.test()

module mbas002
using Test
using SkylineLDLT: update_skyline!, diagonal_addresses
function test()
    dofnums = [3 4; 7 8; 11 12; 9 10; 5 6; 1 2]
    bars = [1 2; 1 6; 6 5; 5 2; 6 2; 2 3; 3 4; 2 4]
    mht = fill(0, maximum(dofnums[:]))
    for b in 1:size(bars, 1)
        update_skyline!(mht, [d for d in dofnums[bars[b, :], :]])
    end
    # @show mht
    @test mht == [0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5]
    maxa = diagonal_addresses(mht)
    @test maxa == [1, 2, 4, 7, 11, 16, 22, 29, 37, 40, 44, 49, 55] 
    true
end
end
using .mbas002
mbas002.test()

module mbas004
using Test
using SkylineLDLT: SkylineMatrix, findnz
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
using .mbas004
mbas004.test()

module mbas004a
using Test
using SkylineLDLT: SkylineMatrix, sparse
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
using .mbas004a
mbas004a.test()

module mbas005
using Test
using LinearAlgebra
using SkylineLDLT: SkylineMatrix, ldlt_factorize!, ldlt_solve
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
    ldlt_factorize!(sky)
    F = Matrix(sparse(sky; symm = false))
     D = tril(triu(F, 0), 0)
     Lt = triu(F, 1) + LinearAlgebra.I
     @test norm(Lt' * D * Lt -  A) < 1e-6 * norm(A)
    true
end
end
using .mbas005
mbas005.test()

module mbas005a
using Test
using LinearAlgebra
using SkylineLDLT: SkylineMatrix, ldlt_factorize!, ldlt_solve
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
    ldlt_factorize!(sky)
    F = Matrix(sparse(sky; symm = false))
     D = tril(triu(F, 0), 0)
     Lt = triu(F, 1) + LinearAlgebra.I
     @test norm(Lt' * D * Lt -  A) < 1e-6 * norm(A)
    true
end
end
using .mbas005a
mbas005a.test()

module mbas005b
using Test
using LinearAlgebra
using SkylineLDLT: SkylineMatrix, ldlt_factorize!, ldlt_solve
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
    ldlt_factorize!(sky)
    F = Matrix(sparse(sky; symm = false))
     D = tril(triu(F, 0), 0)
     Lt = triu(F, 1) + LinearAlgebra.I
     @test norm(Lt' * D * Lt -  A) < 1e-6 * norm(A)
     b = rand(size(A, 1))
    x = ldlt_solve(sky, b)
@test norm(A \ b - x) / norm(x) < 1e-6
    true
end
end
using .mbas005b
mbas005b.test()


module mbas005c
using Test
using LinearAlgebra
using SkylineLDLT: SkylineMatrix, ldlt_factorize!, ldlt_solve
using SparseArrays
function test()
    for s in [0.5, 0.2, 0.1, 0.05]
        for M in 17:3:277
            A = sprand(M, M, s)
            A = A + A' + LinearAlgebra.I
            I, J, V = findnz(A)     
            sky = SkylineMatrix(I, J, V, M)
            ldlt_factorize!(sky)
            b = rand(size(A, 1))
            x = ldlt_solve(sky, b)
            @test norm(A \ b - x) / norm(x) < 1e-6
        end
    end
    true
end
end
using .mbas005c
mbas005c.test()

