"""
    SkylineMatrix

Skyline matrix storage of a symmetric matrix. L*D*L^T factorization and solution.

Version 4 developed from scratch.

The storage here is different from STAP. The numbers in each column are stored
from the top row to the bottom. The addresses are for M+1 "first nonzero in a
column".

The indexing is developed to mimic dense-matrix addressing. As a concession to
performance, the row starting index is precomputed for each column.
"""

module Ldlt4

import Base
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

__cs(frli, c) = frli[c+1] - c - 1
_cs(sky, c) = __cs(sky.frli, c)
_li(r, cs) = cs + r
_1strowincol(sky, c) = sky.frli[c] - _cs(sky, c)

Base.IndexStyle(::Type{<:SkylineMatrix}) = IndexLinear()
Base.size(sky::SkylineMatrix) = (sky.dim, sky.dim)
Base.size(sky::SkylineMatrix, which) = size(sky)[which]
Base.getindex(sky::SkylineMatrix, r::Int, cs::Int) = sky.coefficients[cs + r]
Base.getindex(sky::SkylineMatrix, r::UnitRange{IT}, cs::IT) where {IT} =  @views sky.coefficients[cs .+ (r)]
Base.setindex!(sky::SkylineMatrix, v, r::Int, cs::Int) = (sky.coefficients[cs + r] = v)

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
                cs = __cs(frli, c)
                coefficients[_li(r, cs)] += V[i] 
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
        for r in _1strowincol(sky, c):c
            cs = _cs(sky, c)
            v = sky[r, cs]
            if v != zero(T) 
                push!(I, r)
                push!(J, c)
                push!(V, v)
                if r != c && symm
                    push!(I, c)
                    push!(J, r)
                    push!(V, v)
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

function factorize!(F::MT) where {MT<:SkylineMatrix}
    for j in 2:size(F, 2)
        js = _cs(F, j)
        frj = _1strowincol(F, j)
        for i in 1:j-1
            frij = max(_1strowincol(F, i), frj)
            if frij <= i-1
                is = _cs(F, i)
                F[i, js] -= @views dot(F[frij:i-1, is], F[frij:i-1, js])
            end
        end
        s = F[j, js]
        for r in frj:j-1
            rs = _cs(F, r)
            t = F[r, js] / F[r, rs]
            s -= t*F[r, js];
            F[r, js] = t
        end
        F[j, js] = s
    end
end
 
function solve(F::MT, rhs) where {MT<:SkylineMatrix}
    M = size(F, 1)
    x = fill(0.0, length(rhs))
    # Solve L * z = b
    b = deepcopy(rhs)
    z = deepcopy(x)
    z[1] = b[1]
    for j in 2:M
        js = _cs(F, j)
        s = 0.0
        for k in _1strowincol(F, j):j-1
            s += F[k, js] * z[k]
        end
        z[j] = (b[j] - s)
    end
    # Solve L' * x = D^-1 * z
    for j in 1:M
        js = _cs(F, j)
        z[j] /= F[j, js]
    end
    x[M] = z[M]
    for j in M-1:-1:1
        s = 0.0
        for k in j+1:M 
            r = _1strowincol(F, k)
            if j >= r
                ks = _cs(F, k)
                s += F[j, ks] * x[k]
            end
        end
        x[j] = (z[j] - s) 
    end
    return x
end

end # module