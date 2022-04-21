"""
    SkylineMatrix

Skyline matrix storage of a symmetric matrix. L*D*L^T factorization and solution.

Version 2 developed from scratch.

The storage here is different from STAP. The numbers in each column are stored
from the top row to the bottom. The addresses are for M+1 "first nonzero in a
column".

"""

module Ldlt2

import Base: size
using LinearAlgebra
import SparseArrays: findnz, nnz
import SparseArrays: sparse

struct SkylineMatrix{IT, T}
    dim::Int64 # dimension of the square matrix
    # Linear index of first row in a given column. The (dim+1) address is for an
    # extra column and gives the (total number of stored entries + 1).
    frli::Vector{IT} 
    coefficients::Vector{T} # stored entries of the matrix
end

size(A::SkylineMatrix) = (A.dim, A.dim)

function SkylineMatrix(frli::Vector{IT}, z = zero(T)) where {IT, T}
    dim = length(frli)
    coefficients = fill(z, frli[end])
    return SkylineMatrix(dim, frli, coefficients)
end

function _update_skyline!(column_heights, dofnums)
    minr = minimum(dofnums)
    for c in dofnums
        h = c - minr + 1
        if column_heights[c] < h
            column_heights[c] = h
        end
    end
    return column_heights
end

function _diagonal_addresses(column_heights)
    d = fill(0, length(column_heights))
    d .= column_heights
    d[1] = 1
    for i in 2:length(d)
        d[i] = d[i-1] + d[i]
    end
    return d
end

nnz(A::SkylineMatrix{IT, T}) where {IT, T} = A.frli[end] - 1

_cs(frli, c) = frli[c+1] - c - 1
idx(frli, r, c) = _cs(frli, c) + r
li(r, cs) = cs + r
li(r::UnitRange{Int64}, cs) = cs .+ r
firstr(frli, c) = c - (frli[c+1] - frli[c]) + 1 

function SkylineMatrix(I::Vector{IT}, J::Vector{IT}, V::Vector{T}, m) where {IT, T}
    dim = m
    column_heights = fill(zero(IT), m)
    for i in 1:length(I)
        r = I[i]; c = J[i]
        if r != 0 && c != 0
            _update_skyline!(column_heights, (r, c))
        end
    end
    frli = vcat([1], _diagonal_addresses(column_heights) .+ 1)
    coefficients = fill(zero(T), frli[end]-1)
    for i in 1:length(I)
        r = I[i]; c = J[i]
        if r != 0 && c != 0
            if c >= r
                cs = _cs(frli, c)
                coefficients[li(r, cs)] += V[i] 
            end
        end
    end
    return SkylineMatrix(dim, frli, coefficients)
end


function findnz(sky::SkylineMatrix{IT, T}; symm = true) where {IT, T}
    frli = sky.frli
    I = IT[]
    J = IT[]
    V = T[]
    for c in 1:sky.dim
        for r in firstr(frli, c):c
            lk = idx(frli, r, c)
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

function _inner_factorize!(M, F, frli)
    for j in 2:M
        js = _cs(frli, j)
        frj = firstr(frli, j)
        @inbounds for i in 1:j-1
            fri = firstr(frli, i)
            frij = max(fri, frj)
            if frij <= i-1
                is = _cs(frli, i)
                F[li(i, js)] -= @views dot(F[li(frij:i-1, is)], F[li(frij:i-1, js)])
            end
        end
        s = F[li(j, js)]
        for r in frj:j-1
            rs = _cs(frli, r)
            t = F[li(r, js)] / F[li(r, rs)]
            s -= t*F[li(r, js)];
            F[li(r, js)] = t
        end
        F[li(j, js)] = s
    end
end

function factorize!(A::MT) where {MT<:SkylineMatrix}
    _inner_factorize!(A.dim, A.coefficients, A.frli)
end
 
function solve(A::MT, rhs) where {MT<:SkylineMatrix}
    F = A.coefficients
    frli = A.frli
    M = A.dim
    x = fill(0.0, length(rhs))
    # Solve L * z = b
    b = deepcopy(rhs)
    z = deepcopy(x)
    z[1] = b[1]
    for j in 2:M
        s = 0.0
        for k in firstr(frli, j):j-1
            s += F[idx(frli, k, j)] * z[k]
        end
        z[j] = (b[j] - s)
    end
    # Solve L' * x = D^-1 * z
    for j in 1:M
        z[j] /= F[idx(frli, j, j)]
    end
    x[M] = z[M]
    for j in M-1:-1:1
        s = 0.0
        for k in j+1:M 
            r = firstr(frli, k)
            if j >= r
                s += F[idx(frli, j, k)] * x[k]
            end
        end
        x[j] = (z[j] - s) 
    end
    return x
end

end # module