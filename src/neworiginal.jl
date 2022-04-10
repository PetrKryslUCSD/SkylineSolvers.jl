"""
    SkylineMatrix

Skyline matrix storage of a symmetric matrix. L*D*L^T factorization and solution.

Version developed from scratch.

The storage here is different from STAP. The numbers in each column are stored
from the top row to the bottom. The diagonal addresses are only for the M
columns (not for M+1 addresses as in STAP).
"""

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

idx(das, r, c) = das[c] + r - c

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
                lk = idx(das, r, c)
                coefficients[lk] += V[i] 
            end
        end
    end
    return SkylineMatrix(dim, das, coefficients)
end

firstr(das, c) = (c > 1 ? c - (das[c] - das[c-1]) + 1 : 1)

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

    # A = [                                                                       
    #   2.50000e+00  -1.48222e-02  -6.84521e-02  -6.15827e-02
    #  -1.48222e-02   2.38645e+00  -8.83376e-02   0.00000e+00
    #  -6.84521e-02  -8.83376e-02   2.50000e+00  -4.13988e-02
    #  -6.15827e-02   0.00000e+00  -4.13988e-02   2.50000e+00 
    # ]

    # A = [                                                                       
    #   5.0 -4.0  1.0 0.0
    #  -4.0 6.0 -4.0 1.0
    #  1.0 -4.0 6.0 -4.0
    #  0.0 1.0 -4.0 5.0
    # ]

    # A = [
    # 2.0 -2.0 0.0 0.0 -1.0
    # -2.0 3.0 -2.0 0.0 0.0
    # 0.0 -2.0 5.0 -3.0 0.0
    # 0.0 0.0 -3.0 10.0 4.0
    # -1.0 0.0 0.0 4.0 10.0
    # ]

function print_matrix(A)
    for k in 1:size(A, 1)
        println(A[k,:])
    end
end


# function ldlt1(A)
#     M = size(A, 1)
#     F = deepcopy(A)
#     for col in 1:M-1
#         @show col
#         piv = F[col,col];
#         col1 = col+1;
#         for row in col1:M
#             mul = -F[row,col]/piv;
#             for k in col1:M
#                 F[row,k] = F[row,k]+mul*F[col,k];
#             end
#             F[row,col] *= -1;
#         end
#         print_matrix(F)
#     end
#     F
#     triu(F, 0)
# end

# Version of ldlt1 that uses the skyline
# This function produces an L*D*L^T factorization in-place so that the upper
# triangle of the matrix contains the elements of the L matrix and the D matrix
# is on the diagonal.
# A = [
# 2.0 -2.0 0.0 0.0 -1.0
# -2.0 3.0 -2.0 0.0 0.0
# 0.0 -2.0 5.0 -3.0 0.0
# 0.0 0.0 -3.0 10.0 4.0
# -1.0 0.0 0.0 4.0 10.0
# ]
# using LinearAlgebra
# F = ldlt2(A)
# D = tril(triu(F, 0), 0)
# Lt = triu(F, 1) + I
# L = Lt'
# L * D * L' -  A
function ldlt2(A)
    M = size(A, 1)
    F = deepcopy(triu(A, 0))
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

# b = [5.77658e-01
#  4.11267e-01
#  8.53042e-01
#  3.50732e-02
#  8.20537e-01]
# z = L \ b    
# x = L' \ (D \ z)  
# A \ b      

function ldlt2_solve(F, b)
    M = size(F, 1)
    x = fill(0.0, length(b))
    # Solve L * z = b
    rhs = deepcopy(b)
    z = deepcopy(x)
    z[1] = b[1]
    for j in 2:M
        s = 0.0
        for k in 1:j-1
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
            s += F[j, k] * x[k]
        end
        x[j] = (z[j] - s) 
    end
    x
end

# function chol1sol(R, b)
#     @show R \ b
#     x = deepcopy(b)
#     for k in 1:M
#         t = 0.0
#         for j in 1:k-1
#             t -= R[k, j] * x[j]
#         end
#         x[k] = (b[k] + t) / R[k, k]
#     end
#     @show x
#     @show R' \ x
#     for k in M:-1:1
#         t = x[k]
#         for j in k+1:M
#             t -= R[j, k] * x[j]
#         end
#         x[k] = t / R[k, k]
#     end
#     @show x
# end

function _inner_ldlt_factorize!(M, F, das)
    for j in 2:M
        for i in 1:j-1
            iij = idx(das, i, j)
            s = F[iij]
            mij = max(firstr(das, i), firstr(das, j))
            if mij <= i-1
                ii = idx(das, mij, i)
                jj = idx(das, mij, j)
                s -= dot(view(F, ii.+(0:1:(i-1-mij))), view(F, jj.+(0:1:(i-1-mij))))
            end
            # for r in mij:i-1
            #     s -= F[idx(das, r, i)]*F[idx(das, r, j)];
            # end
            F[iij] = s
        end
        s = F[idx(das, j, j)]
        @inbounds for r in firstr(das, j):j-1
            t = F[idx(das, r, j)] / F[idx(das, r, r)]
            s -= t*F[idx(das, r, j)];
            F[idx(das, r, j)] = t
        end
        F[idx(das, j, j)] = s
    end
end

function ldlt_factorize!(A::MT) where {MT<:SkylineMatrix}
    V = A.coefficients
    das = A.das
    _inner_ldlt_factorize!(A.dim, V, das)
end
 
function ldlt_solve(A::MT, rhs) where {MT<:SkylineMatrix}
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

