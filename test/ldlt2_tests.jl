
module mldlt2005
using Test
using LinearAlgebra
using SkylineSolvers.Ldlt2: SkylineMatrix, factorize!, solve
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
using .mldlt2005
mldlt2005.test()

module mldlt2001
using Test
using SkylineSolvers.Ldlt2: _update_skyline!

function test()
    dofnums = [3 4; 7 8; 11 12; 9 10; 5 6; 1 2]
    bars = [1 2; 1 6; 6 5; 5 2; 6 2; 2 3; 3 4; 2 4]
    skylngs = fill(0, maximum(dofnums[:]))
    for b in 1:size(bars, 1)
        _update_skyline!(skylngs, [d for d in dofnums[bars[b, :], :]])
    end
    # @show skylngs
    @test skylngs == [1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6]
    true
end
end
using .mldlt2001
mldlt2001.test()

module mldlt2002
using Test
using SkylineSolvers.Ldlt2: _update_skyline!, _diagonal_addresses
function test()
    dofnums = [3 4; 7 8; 11 12; 9 10; 5 6; 1 2]
    bars = [1 2; 1 6; 6 5; 5 2; 6 2; 2 3; 3 4; 2 4]
    skylngs = fill(0, maximum(dofnums[:]))
    for b in 1:size(bars, 1)
        _update_skyline!(skylngs, [d for d in dofnums[bars[b, :], :]])
    end
    # @show skylngs
    @test skylngs == [1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6]
    d = _diagonal_addresses(skylngs)
    @test d == [1, 3, 6, 10, 15, 21, 28, 36, 39, 43, 48, 54]  
    true
end
end
using .mldlt2002
mldlt2002.test()

module mldlt2004
using Test
using SkylineSolvers.Ldlt2: SkylineMatrix, findnz
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
using .mldlt2004
mldlt2004.test()

module mldlt2004a
using Test
using SkylineSolvers.Ldlt2: SkylineMatrix, sparse
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
using .mldlt2004a
mldlt2004a.test()

module mldlt2005a
using Test
using LinearAlgebra
using SkylineSolvers.Ldlt2: SkylineMatrix, factorize!, solve
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
using .mldlt2005a
mldlt2005a.test()

module mldlt2005b
using Test
using LinearAlgebra
using SkylineSolvers.Ldlt2: SkylineMatrix, factorize!, solve
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
using .mldlt2005b
mldlt2005b.test()


module mldlt2005c
using Test
using LinearAlgebra
using SkylineSolvers.Ldlt2: SkylineMatrix, factorize!, solve
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
using .mldlt2005c
mldlt2005c.test()

