"""
    SkylineMatrix

Skyline matrix storage of a symmetric matrix. L*D*L^T factorization and solution.

Version 5 developed from scratch.

The entries are ordered column-by-column, from 
    the first non zero entry in each column down to the diagonal.
    This array records the linear index of the diagonal 
    entry, `das[k]`, in a given column `k`. 
    The `das[0]` address is for an dummy 0-th column.

The indexing is developed to mimic dense-matrix addressing.
"""

module Ldlt5

import Base
using LinearAlgebra
import SparseArrays: findnz, nnz
import SparseArrays: sparse
using OffsetArrays

struct SkylineMatrix{IT, T}
    # Dimension of the square matrix
    dim::IT 
    # The entries are ordered column-by-column, from 
    # the first non zero entry in each column down to the diagonal.
    # This array records the linear index of the diagonal 
    # entry, `das[k]`, in a given column `k`. 
    # The `das[0]` address is for an dummy 0-th column.
    das::OffsetVector{IT, Vector{IT}}
    # Stored entries of the matrix
    coefficients::Vector{T} 
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
    return OffsetArray(vcat([0], cumsum(column_heights)), 0:length(column_heights))
end

nnz(A::SkylineMatrix{IT, T}) where {IT, T} = A.das[end] 

_li(r, cs) = cs + r
_cs(das, c) = das[c] - c
_1strowincol(sky, c) = c - (sky.das[c] - sky.das[c-1]) + 1 

Base.IndexStyle(::Type{<:SkylineMatrix}) = IndexLinear()
Base.size(sky::SkylineMatrix) = (sky.dim, sky.dim)
Base.size(sky::SkylineMatrix, which) = size(sky)[which]
Base.getindex(sky::SkylineMatrix, r::Int, c::Int) = sky.coefficients[_cs(sky.das, c) + r]
Base.getindex(sky::SkylineMatrix, r::UnitRange{IT}, c::IT) where {IT} =  @views sky.coefficients[_cs(sky.das, c) .+ (r)]
Base.setindex!(sky::SkylineMatrix, v, r::Int, c::Int) = (sky.coefficients[_cs(sky.das, c) + r] = v)

function SkylineMatrix(I::Vector{IT}, J::Vector{IT}, V::Vector{T}, m) where {IT, T}
    dim = m
    column_heights = fill(zero(IT), m)
    for i in 1:length(I)
        r = I[i]; c = J[i]
        if r != 0 && c != 0
            _update_skyline!(column_heights, (r, c))
        end
    end
    das = _diagonal_addresses(column_heights)
    coefficients = fill(zero(T), das[end])
    for i in 1:length(I)
        r = I[i]; c = J[i]
        if r != 0 && c != 0
            if c >= r
                cs = _cs(das, c)
                coefficients[_li(r, cs)] += V[i] 
            end
        end
    end
    return SkylineMatrix(dim, das, coefficients)
end

function findnz(sky::SkylineMatrix{IT, T}; symm = true) where {IT, T}
    I = IT[]
    J = IT[]
    V = T[]
    for c in 1:sky.dim
        for r in _1strowincol(sky, c):c
            v = sky[r, c]
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
        frj = _1strowincol(F, j)
        for i in frj:j-1
            frij = max(_1strowincol(F, i), frj)
            if frij <= i-1
                F[i, j] -= @views dot(F[frij:i-1, i], F[frij:i-1, j])
            end
        end
        s = F[j, j]
        for r in frj:j-1
            t = F[r, j] / F[r, r]
            s -= t*F[r, j];
            F[r, j] = t
        end
        F[j, j] = s
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
        s = 0.0
        for k in _1strowincol(F, j):j-1
            s += F[k, j] * z[k]
        end
        z[j] = (b[j] - s)
    end
    # Solve L' * x = D^-1 * z
    for j in 1:M
        z[j] /= F[j, j]
    end
    x[M] = z[M]
    for j in M-1:-1:1
        s = 0.0
        for k in j+1:M 
            r = _1strowincol(F, k)
            if j >= r
                s += F[j, k] * x[k]
            end
        end
        x[j] = (z[j] - s) 
    end
    return x
end

end # module