module Chol

import Base: size
using LinearAlgebra
import SparseArrays: findnz, nnz
import SparseArrays: sparse

struct SkylineMatrix{IT, T}
    dim::Int64
    das::Vector{IT}
    coefficients::Vector{T}
end

function update_skyline!(skylngs, dofnums)
    ndof = length(dofnums)
    for i in 1:ndof
        gi = dofnums[i]
        if gi != 0 
            w = 1
            for j in 1:ndof
                gj = dofnums[j]
                if gj != 0 
                    im = dofnums[i]-dofnums[j]+1
                    if im > w
                        w = im
                    end
                end
            end
            if w > skylngs[gi]
                skylngs[gi] = w
            end
        end
    end
    return skylngs
end

function diagonal_addresses(skylngs)
    d = fill(0, length(skylngs))
    d .= skylngs
    for i in 2:length(d)
        d[i] = d[i] + d[i-1]
    end
    return d
end

nnz(A::SkylineMatrix{IT, T}) where {IT, T} = A.das[end]

function SkylineMatrix(I::Vector{IT}, J::Vector{IT}, V::Vector{T}, m) where {IT, T}
    dim = m
    skylngs = fill(zero(IT), m)
    for i in 1:length(I)
        r = I[i]; c = J[i]
        if r != 0 && c != 0
            update_skyline!(skylngs, (r, c))
        end
    end
    das = diagonal_addresses(skylngs)
    coefficients = fill(zero(T), das[end])
    for i in 1:length(I)
        r = I[i]; c = J[i]
        if r != 0 && c != 0
            if c <= r
                idx = das[r]-(r-c)
                coefficients[idx] += V[i] 
            end
        end
    end
    return SkylineMatrix(dim, das, coefficients)
end

function findnz(A::SkylineMatrix{IT, T}) where {IT, T}
    I = IT[]
    J = IT[]
    V = T[]
    r = 1
    rw = 1
    idx = A.das[r]
    if A.coefficients[idx] != zero(T) 
        push!(I, r)
        push!(J, r)
        push!(V, A.coefficients[idx])
    end
    for r in 2:length(A.das)
        rw = A.das[r] - A.das[r-1]
        for c in r-rw+1:r
            idx = A.das[r]-(r-c)
            if A.coefficients[idx] != zero(T) 
                push!(I, r)
                push!(J, c)
                push!(V, A.coefficients[idx])
                if r != c
                    push!(I, c)
                    push!(J, r)
                    push!(V, A.coefficients[idx])
                end
            end
        end
    end
    return I, J, V
end


# function _inner_cholesky_factorize_2!(dim, V, das)
#     rw = fill(0, dim)
#     rw[1] = 1
#     for r in 2:dim
#         rw[r] = das[r]-das[r-1]
#     end
#     pc = fill(0, dim)
#     pc[1] = 0
#     for r in 2:dim
#         pc[r] = das[r]-r
#     end
#     z = zero(eltype(V))
#     V[1] = sqrt(V[1])
#     for r in 2:dim
#         kr = pc[r]
#         l = das[r-1]-kr+1
#         x = z
#         for c in l:r
#             x = V[kr+c]
#             kc = pc[c]
#             if c != 1
#                 ll = das[c-1]-kc+1
#                 ll = max(l,ll)
#                 if ll != c
#                     @inbounds for  k in ll:(c-1)
#                         x -= V[kr+k]*V[kc+k] 
#                     end
#                     # x -= dot(view(V, kr+ll:kr+(c-1)), view(V, kc+ll:kc+(c-1)))
#                 end
#             end
#             V[kr+c] = x/V[kc+c]
#         end
#         V[kr+r] = sqrt(x)
#     end
# end

# function _inner_cholesky_factorize_3!(dim, V, das)
#     rw = fill(0, dim)
#     rw[1] = 1
#     for r in 2:dim
#         rw[r] = das[r]-das[r-1]
#     end
#     @show maxrw = maximum(rw)
#     z = zero(eltype(V))
#     Vrbuffer = fill(z, maxrw+1)

#     V[1] = sqrt(V[1])
#     for r in 2:dim
#         @show kr = das[r]-r
#         @show l = das[r-1]-kr+1
#         copyto!(Vrbuffer, view(V, kr+1:kr+r))
#         println("done") 
#         x = z
#         @show l:r
#         for c in l:r
#             @show c
#             x = Vrbuffer[c] # x = V[kr+c]
#             @assert x == V[kr+c]
#             @show Vrbuffer[1:r]
#             kc = das[c]-c
#             if c != 1
#                 ll = das[c-1]-kc+1
#                 ll = max(l,ll)
#                 if ll != c
#                     @show V[kr+ll:kr+(c-1)], Vrbuffer[ll:(c-1)]
#                     x -= dot(view(Vrbuffer, ll:(c-1)), view(V, kc+ll:kc+(c-1)))
#                     #x -= dot(view(V, kr+ll:kr+(c-1)), view(V, kc+ll:kc+(c-1)))
#                 end
#             end
#             V[kr+c] = x/V[kc+c]
#         end
#         V[kr+r] = sqrt(x)
#     end
# end

function _inner_factorize!(dim, V, das)
    z = zero(eltype(V))
    V[1] = sqrt(V[1])
    for r in 2:dim
        kr = das[r]-r
        l = das[r-1]-kr+1
        x = z
        for c in l:r
            x = V[kr+c]
            kc = das[c]-c
            if c != 1
                ll = das[c-1]-kc+1
                ll = max(l,ll)
                if ll != c
                    # @inbounds for  k in ll:(c-1)
                    #     x -= V[ki+k]*V[kc+k] 
                    # end
                    x -= dot(view(V, kr+ll:kr+(c-1)), view(V, kc+ll:kc+(c-1)))
                end
            end
            V[kr+c] = x/V[kc+c]
        end
        V[kr+r] = sqrt(x)
    end
end

function factorize!(A::MT) where {MT<:SkylineMatrix}
    V = A.coefficients
    das = A.das
    _inner_factorize!(A.dim, V, das)
end

function solve(A::MT, rhs) where {MT<:SkylineMatrix}
    n=A.dim
    V = A.coefficients
    das = A.das
    b = deepcopy(rhs)
    b[1] = b[1]/V[1]
    for i in 2:A.dim
        ki = das[i]-i
        l = das[i-1]-ki+1 
        x = b[i]
        if l != i
            m = i-1
            for j in l:m 
                x = x-V[ki+j]*b[j]
            end
        end
        b[i] = x/V[ki+i]
    end
    for it in 2:n
        i = n+2-it
        ki = das[i]-i
        x = b[i]/V[ki+i]
        b[i] = x
        l = das[i-1]-ki+1
        if l != i
            m = i-1
            for k in l:m
                b[k] -= x*V[ki+k]
            end
        end
    end
    b[1] = b[1]/V[1]
    return b
end


end # module