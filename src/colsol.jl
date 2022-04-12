"""
    SkylineMatrix

Skyline matrix storage of a symmetric matrix. L*D*L^T factorization and solution.

Version adapted from https://github.com/HaoguangYang/OpenSTAP
"""
module Colsol

import Base: size
using LinearAlgebra
import SparseArrays: findnz, nnz
import SparseArrays: sparse

struct SkylineMatrix{IT, T}
    dim::Int64
    maxa::Vector{IT}
    coefficients::Vector{T}
end

size(A::SkylineMatrix) = (A.dim, A.dim)

function SkylineMatrix(maxa::Vector{IT}, z = zero(T)) where {IT, T}
    @show dim = length(maxa) - 1
    @show maxa[end]
    coefficients = fill(z, maxa[end])
    return SkylineMatrix(dim, maxa, coefficients)
end

function update_skyline!(mht, lm)
    nd = length(lm)
    ls=typemax(eltype(mht)) 
    for i=1:nd
        if lm[i] != 0
            if lm[i]-ls < 0
                ls=lm[i]
            end
        end
    end 
    for i=1:nd
        ii=lm[i]
        if ii != 0
            me=ii - ls
            if me > mht[ii]
                mht[ii]=me
            end
        end 
    end 
end

function diagonal_addresses(mht)
    neq = length(mht)
    nn=neq + 1
    maxa = fill(0, nn)
    for i=1:nn
        maxa[i]=0.0
    end 
    maxa[1]=1
    maxa[2]=2
    if neq > 1 
        for  i=2:neq
            maxa[i+1]=maxa[i] + mht[i] + 1
        end 
    end 
    nwk=maxa[neq+1] - maxa[1]
    return maxa
end

nnz(A::SkylineMatrix{IT, T}) where {IT, T} = A.maxa[end]-1

idx(maxa, r, c) = maxa[c] + c - r

function SkylineMatrix(I::Vector{IT}, J::Vector{IT}, V::Vector{T}, m) where {IT, T}
    dim = m
    mht = fill(zero(IT), m)
    for i in 1:length(I)
        r = I[i]; c = J[i]
        if r != 0 && c != 0
            update_skyline!(mht, (r, c))
        end
    end
    maxa = diagonal_addresses(mht)
    coefficients = fill(zero(T), maxa[end]-1)
    for i in 1:length(I)
        r = I[i]; c = J[i]
        if r != 0 && c != 0
            if c >= r
                lk = idx(maxa, r, c)
                coefficients[lk] += V[i] 
            end
        end
    end
    return SkylineMatrix(dim, maxa, coefficients)
end

firstr(maxa, c) = -maxa[c+1] + maxa[c] + c + 1

function findnz(sky::SkylineMatrix{IT, T}; symm = true) where {IT, T}
    maxa = sky.maxa
    I = IT[]
    J = IT[]
    V = T[]
    for c in 1:sky.dim
        for r in firstr(maxa, c):c
            lk = idx(maxa, r, c)
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

function print_matrix(A)
    for k in 1:size(A, 1)
        println(A[k,:])
    end
end

function _inner_factorize!(M, a, maxa)
    nn=length(maxa) - 1
    for n=1:nn
        kn=maxa[n]
        kl=kn + 1
        ku=maxa[n+1] - 1
        kh=ku - kl
        if (kh > 0) 
            k=n - kh
            ic=0
            klt=ku
            for j=1:kh
                ic=ic + 1
                klt=klt - 1
                ki=maxa[k]
                nd=maxa[k+1] - ki - 1
                if (nd  > 0) 
                    kk=min(ic,nd)
                    c=0.
                    for l=1:kk
                        c=c + a[ki+l]*a[klt+l]
                    end
                    a[klt]=a[klt] - c
                end 
                k=k + 1
            end
        end
        if (kh >= 0) 
            k=n
            b=0.0
            for kk=kl:ku
                k=k - 1
                ki=maxa[k]
                c=a[kk]/a[ki]
                b=b + c*a[kk]
                a[kk]=c
            end
            a[kn]=a[kn] - b
        end
        if (a[kn] == 0) 
            @error "Zero Pivot"
        end 
    end
end

function factorize!(sky::MT) where {MT<:SkylineMatrix}
    _inner_factorize!(sky.dim, sky.coefficients, sky.maxa)
end

function _inner_ldlt_solve(dim, a, maxa, v)
    nn=length(maxa) - 1
    for n=1:nn
        kl=maxa[n] + 1
        ku=maxa[n+1] - 1
        if (ku-kl  >=  0) 
            k=n
            c=0.0
            for kk=kl:ku
                k=k - 1
                c=c + a[kk]*v[k]
            end 
            # c = dot_product(a(kl:ku),v(n-1:n-(ku-kl)-1:-1))
            v[n]=v[n] - c
        end 
    end 
    # back-substitute
    for n=1:nn
       k=maxa[n]
       v[n]=v[n]/a[k]
    end 
    (nn == 1) && return
    n=nn
    for l=2:nn
        kl=maxa[n] + 1
        ku=maxa[n+1] - 1
        if (ku-kl  >= 0) 
            k=n
            for kk=kl:ku
                k=k - 1
                v[k]=v[k] - a[kk]*v[n]
            end 
        end
        n=n - 1
    end 
    return v
end

 
function solve(sky::MT, rhs) where {MT<:SkylineMatrix}
    x = fill(0.0, length(rhs))
    x .= rhs
    return _inner_ldlt_solve(sky.dim, sky.coefficients, sky.maxa, x)
end

end # module