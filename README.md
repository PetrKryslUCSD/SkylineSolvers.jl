# SkylineSolvers.jl

*The package is under current development.*

## Usage

The skyline matrices are currently created from the "coordinate" representation of a
sparse matrix.
```
using SkylineSolvers, DataDrop
K = DataDrop.retrieve_matrix("K.h5")
@show size(K)
I, J, V = findnz(K)     
sky = SkylineSolvers.Ldlt2.SkylineMatrix(I, J, V, size(K, 1))
```

The two main operations are "factorize" and "solve":
```
SkylineSolvers.Ldlt2.Factorize!(sky)
b = rand(size(A, 1))
x = SkylineSolvers.Ldlt2.solve(sky, b)
@test norm(A \ b - x) / norm(x) < 1e-6
```

At the moment, after factorization `sky` holds the factorized matrix, but there
are no functions to extract the individual factors.

## Notes

- The package is divided into several modules.
- The package is intended for symmetric indefinite matrices (except the Cholesky
  decomposition requires a positive definite matrix).
- The modules define the type `SkylineMatrix`, which are mutually incompatible.
  In each module the matrix is stored under a skyline, and only one half of the
  matrix is actually stored.
- The module `Chol` defines a Cholesky decomposition and triangular solve.
- The modules `Ldlt`, `Ldlt2`, `Ldlt3` define a LDLT decomposition and
  triangular solve each. The module `Ldlt3` is most pleasing aesthetically, but
  it is clearly the slowest. `Ldlt`, `Ldlt2` are roughly equally fast.
- The module `Colsol` defines the original skyline solution from the textbook of
  KJ Bathe. It is clearly around three times slower then  `Ldlt` or `Ldlt2`.
- No renumbering is undertaken in order to minimize the number of entries store
  below the skyline. If the matrix is numbered in an unfortunate way, used the
  package `SymRCM` to reorder the matrix first.