
module mldlt3005
using Test
using LinearAlgebra
using SkylineSolvers.Ldlt3: SkylineMatrix, factorize!, solve
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
using .mldlt3005
mldlt3005.test()

module mldlt3001
using Test
using SkylineSolvers.Ldlt3: _update_skyline!

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
using .mldlt3001
mldlt3001.test()

module mldlt3002
using Test
using SkylineSolvers.Ldlt3: _update_skyline!, _diagonal_addresses
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
using .mldlt3002
mldlt3002.test()

module mldlt3004
using Test
using SkylineSolvers.Ldlt3: SkylineMatrix, findnz
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
using .mldlt3004
mldlt3004.test()

module mldlt3004a
using Test
using SkylineSolvers.Ldlt3: SkylineMatrix, sparse
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
using .mldlt3004a
mldlt3004a.test()

module mldlt3005a
using Test
using LinearAlgebra
using SkylineSolvers.Ldlt3: SkylineMatrix, factorize!, solve
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
using .mldlt3005a
mldlt3005a.test()

module mldlt3005b
using Test
using LinearAlgebra
using SkylineSolvers.Ldlt3: SkylineMatrix, factorize!, solve
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
using .mldlt3005b
mldlt3005b.test()


module mldlt3005c
using Test
using LinearAlgebra
using SkylineSolvers.Ldlt3: SkylineMatrix, factorize!, solve
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
using .mldlt3005c
mldlt3005c.test()


module mldlt3005d
using Test
using LinearAlgebra
using SkylineSolvers.Ldlt3: SkylineMatrix, factorize!, solve
using SparseArrays
function test()
    total = 0
    failed = 0
    for s in [0.8, 0.5,  ]
        for M in 17:3:135
            A = sprand(Float32, M, M, s)
            A = A + A' + LinearAlgebra.I
            I, J, V = findnz(A)     
            sky = SkylineMatrix(I, J, Float32.(V), M)
            factorize!(sky)
            b = rand(Float32, size(A, 1))
            x = solve(sky, b)
            failed = norm((A * x) - b) / norm(b) < 1e-4 ? 0 : 1
            total += 1
        end
    end
    @test (total - failed) > failed
    true
end
end
using .mldlt3005d
mldlt3005d.test()

module mldlt3005e
using Test
using LinearAlgebra
using SkylineSolvers.Ldlt3: SkylineMatrix, factorize!, solve
using SparseArrays
function test()
    total = 0
    failed = 0
    for s in [0.8, 0.5,  ]
        for M in 17:3:135
            A = sprand(Float32, M, M, s)
            A = A + A' + LinearAlgebra.I
            I, J, V = findnz(A)     
            sky = SkylineMatrix(Int32.(I), Int32.(J), Float32.(V), M)
            factorize!(sky)
            b = rand(Float32, size(A, 1))
            x = solve(sky, b)
            failed = norm((A * x) - b) / norm(b) < 1e-4 ? 0 : 1
            total += 1
        end
    end
    @test (total - failed) > failed
    true
end
end
using .mldlt3005e
mldlt3005e.test()
