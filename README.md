# SparseFFT

[![Build Status](https://travis-ci.org/klho/SparseFFT.jl.svg?branch=master)](https://travis-ci.org/klho/SparseFFT.jl)

[![Coverage Status](https://coveralls.io/repos/klho/SparseFFT.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/klho/SparseFFT.jl?branch=master)

[![codecov.io](http://codecov.io/github/klho/SparseFFT.jl/coverage.svg?branch=master)](http://codecov.io/github/klho/SparseFFT.jl?branch=master)

This Julia package provides functions for computing sparse (or "pruned") fast Fourier transforms (FFTs) in one (1D) and two (2D) dimensions. We say that an FFT is sparse if:

- it maps from a full input domain to a sparse output domain, i.e., only a subset of output indices are required; or
- it maps from a sparse input domain to a full output domain, i.e., only a subset of input indices are nonzero.

Sparse FFTs can be efficiently computed by, taking the first case for concreteness, essentially hand-unrolling a standard FFT decimation then selectively recombining to form only the required outputs. More specifically, write a 1D `n`-point transform as:

```julia
y[i] = sum([w(n)^((i-1)*(j-1))*x[j] for j in 1:n])
```

where `w(p) = exp(-2im*pi/p)` or `exp(2im*pi/p)` depending on the transform direction. Then to compute any set of `k` entries, where we assume for simplicity that `m = n/k` is integral, use the identity:

```julia
y[k*(i1-1)+i2] =
    sum([w(m)^((i1-1)*(j2-1))
             * w(n)^((i2-1)*(j2-1))
             * sum([w(k)^((i2-1)*(j1-1))*x[m*(j1-1)+j2] for j1 = 1:k])
         for j2 = 1:m])
```

for `i1 in 1:m` and `i2 in 1:k`, i.e., do `m` FFTs of size `k` (over `j1`) then sum `m` terms (over `j2`) for each entry. The dual sparse-to-full problem is similar, with both algorithms having `O(n*log(k))` complexity. Sparse FFTs in 2D (and higher dimensions) can be handled via tensor 1D transforms.

For further details, see:

- [F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert. A fast randomized algorithm for the approximation of matrices. Appl. Comput. Harmon. Anal. 25: 335-366, 2008](http://dx.doi.org/10.1016/j.acha.2007.12.002)
- http://www.fftw.org/pruned.html

## Usage

SparseFFT follows the same organizational principle as FFTW, with separate planning and execution routines. The primary planning functions in 1D are:

- `plan_spfft (T, n, idx; plan_fft_optargs)`
- `plan_spbfft(T, n, idx; plan_fft_optargs)`

which produce, respectively, plans for forward and backward transforms, with arguments:

- `T`: underlying FFT `Complex` type (e.g., `Complex128`)
- `n`: full domain size
- `idx`: sparse domain indices
- `plan_fft_optargs`: optional arguments for underlying FFT planner

Corresponding execution functions include:

- `spfft_f2s! (y, plan, x; nb)`
- `spfft_s2f! (y, plan, x; nb)`
- `spbfft_f2s!(y, plan, x; nb)`
- `spbfft_s2f!(y, plan, x; nb)`

which perform full-to-sparse (`f2s`) or sparse-to-full (`s2f`) transforms, as appropriate, using a precomputed plan. The arguments `x` and `y` are nominally complex but can also be real; real input `x` is simply upcasted to `Complex` type `T` for the underlying FFT, while real output `y` just takes the real part. The optional argument `nb` specifies a block size for handling a required transpose in a cache-friendly way.

Optimizations using real FFTs are available for the real-to-complex full-to-sparse and complex-to-full sparse-to-real cases. Planning routines:

- `plan_sprfft (T, n, idx; plan_fft_optargs)`
- `plan_spbrfft(T, n, idx; plan_fft_optargs)`

where now `T <: Real` and `idx` for the backward transform must contain only nonredundant frequencies, i.e., up to index `div(n,2) + 1`. Execution routines:

- `sprfft_f2s! (y, plan, x; nb)`
- `spbrfft_s2f!(y, plan, x; nb)`

In 2D, we have essentially the essentially the same interface, with the following exceptions:

- The generic planner now takes the form `plan_sp*fft(T, n1, n2, idx1, idx2; plan_fft_optargs)`, where the full domain has size `n1 x n2` and the sparse domain is defined as the tensor-product grid between `idx1` and `idx2`.

- For `plan_spbrfft`, `idx1` can only contain indices up to `div(n1,2) + 1`.

## Example

The following example computes a random 100 x 100 subset of the spectrum of a 1000 x 1000 field and compares it to naively computing the full FFT:

```julia
using SparseFFT

T = Complex128
n1 = 1000
n2 = 1000
k1 = 100
k2 = 100
idx1 = randperm(n1)[1:k1]
idx2 = randperm(n2)[1:k2]
x = rand(T, n1, n2)
y = rand(T, k1, k2)

@time f = fft(x)[idx1,idx2]
@time P = plan_spfft(T, n1, n2, idx1, idx2)
@time spfft_f2s!(y, P, x)

vecnorm(f - y)/vecnorm(f)
```

Sample output (with annotations):

```julia
  0.043323 seconds (72 allocations: 15.415 MB)     # fft
  0.000344 seconds (128 allocations: 3.254 MB)     # plan_spfft
  0.025395 seconds (7.71 k allocations: 3.072 MB)  # spfft_f2s!
4.066617809196614e-15                              # vecnorm
```
