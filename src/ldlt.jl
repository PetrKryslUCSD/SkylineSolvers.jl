"""
    SkylineMatrix

Skyline matrix storage of a symmetric matrix. L*D*L^T factorization and solution.

Version developed from scratch.

The storage here is different from STAP. The numbers in each column are stored
from the top row to the bottom. The diagonal addresses are only for the M
columns (not for M+1 addresses as in STAP).
"""

module Ldlt

import Base: size
using LinearAlgebra
import SparseArrays: findnz, nnz
import SparseArrays: sparse

struct SkylineMatrix{IT, T}
    dim::Int64
    das::Vector{IT}
    coefficients::Vector{T}
end

size(A::SkylineMatrix) = (A.dim, A.dim)

function SkylineMatrix(das::Vector{IT}, z = zero(T)) where {IT, T}
    dim = length(das)
    coefficients = fill(z, das[end])
    return SkylineMatrix(dim, das, coefficients)
end

function update_skyline!(column_heights, dofnums)
    minr = minimum(dofnums)
    for c in dofnums
        h = c - minr + 1
        if column_heights[c] < h
            column_heights[c] = h
        end
    end
    return column_heights
end

function diagonal_addresses(column_heights)
    d = fill(0, length(column_heights))
    d .= column_heights
    d[1] = 1
    for i in 2:length(d)
        d[i] = d[i-1] + d[i]
    end
    return d
end

nnz(A::SkylineMatrix{IT, T}) where {IT, T} = A.das[end]

_cs(das, c) = das[c] - c
idx(das, r, c) = _cs(das, c) + r
li(r, cs) = cs + r
li(r::UnitRange{Int64}, cs) = cs .+ r
firstr(das, c) = (c > 1 ? c - (das[c] - das[c-1]) + 1 : 1)

function SkylineMatrix(I::Vector{IT}, J::Vector{IT}, V::Vector{T}, m) where {IT, T}
    dim = m
    column_heights = fill(zero(IT), m)
    for i in 1:length(I)
        r = I[i]; c = J[i]
        if r != 0 && c != 0
            update_skyline!(column_heights, (r, c))
        end
    end
    das = diagonal_addresses(column_heights)
    coefficients = fill(zero(T), das[end])
    for i in 1:length(I)
        r = I[i]; c = J[i]
        if r != 0 && c != 0
            if c >= r
                cs = _cs(das, c)
                coefficients[li(r, cs)] += V[i] 
            end
        end
    end
    return SkylineMatrix(dim, das, coefficients)
end


function findnz(sky::SkylineMatrix{IT, T}; symm = true) where {IT, T}
    das = sky.das
    I = IT[]
    J = IT[]
    V = T[]
    for c in 1:length(das)
        for r in firstr(das, c):c
            lk = idx(das, r, c)
            if sky.coefficients[lk] != zero(T) 
                push!(I, r)
                push!(J, c)
                push!(V, sky.coefficients[lk])
                if r != c && symm
                    push!(I, c)
                    push!(J, r)
                    push!(V, sky.coefficients[lk])
                end
            end
        end
    end
    return I, J, V
end

function sparse(sky::SkylineMatrix{IT, T}; symm = true) where {IT, T}
    M = sky.dim
    I, J, V = findnz(sky; symm = symm)
    return sparse(I, J, V, M, M)
end

function dense_1!(F)
    M = size(F, 1)
    for j in 2:M
        for i in 1:j-1
            s = F[i, j]
            for r in 1:i-1
                s -= F[r, i]*F[r, j];
            end
            F[i, j] = s
        end
        s = F[j, j]
        for r in 1:j-1
            t = F[r, j] / F[r, r]
            s -= t*F[r, j];
            F[r, j] = t
        end
        F[j, j] = s
    end
    F
end

function dense_ldlt!(F)
    M = size(F, 1)
    for j in 2:M
        for i in 1:j-1
            F[i, j] -= dot(F[1:i-1, i], F[1:i-1, j])
        end
        s = F[j, j]
        for r in 1:j-1
            t = F[r, j] / F[r, r]
            s -= t * F[r, j]
            F[r, j] = t
        end
        F[j, j] = s
    end
    F
end

function _inner_factorize!(M, F, das, fr)
    for j in 2:M
        js = _cs(das, j)
        frj = fr[j]
        @inbounds for i in 1:j-1
            mij = max(fr[i], frj)
            if mij <= i-1
                is = _cs(das, i)
                r = mij:i-1
                F[li(i, js)] -= @views dot(F[li(r, is)], F[li(r, js)])
            end
        end
        s = F[li(j, js)]
        for r in frj:j-1
            rs = _cs(das, r)
            t = F[li(r, js)] / F[li(r, rs)]
            s -= t*F[li(r, js)];
            F[li(r, js)] = t
        end
        F[li(j, js)] = s
    end
end

function factorize!(A::MT) where {MT<:SkylineMatrix}
    V = A.coefficients
    das = A.das
    fr = Array{eltype(das), 1}(undef, A.dim)
    fr[1] = 1
    for i in 2:A.dim
        fr[i] = firstr(das, i)
    end
    _inner_factorize!(A.dim, V, das, fr)
end
 
function solve(A::MT, rhs) where {MT<:SkylineMatrix}
    F = A.coefficients
    das = A.das
    M = A.dim
    x = fill(0.0, length(rhs))
    # Solve L * z = b
    b = deepcopy(rhs)
    z = deepcopy(x)
    z[1] = b[1]
    for j in 2:M
        s = 0.0
        for k in firstr(das, j):j-1
            s += F[idx(das, k, j)] * z[k]
        end
        z[j] = (b[j] - s)
    end
    # Solve L' * x = D^-1 * z
    for j in 1:M
        z[j] /= F[idx(das, j, j)]
    end
    x[M] = z[M]
    for j in M-1:-1:1
        s = 0.0
        for k in j+1:M 
            r = firstr(das, k)
            if j >= r
                s += F[idx(das, j, k)] * x[k]
            end
        end
        x[j] = (z[j] - s) 
    end
    return x
end

end # module