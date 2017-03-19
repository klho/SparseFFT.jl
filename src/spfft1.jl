#= src/spfft1.jl
=#

abstract SpFFTPlan1{T,K}

immutable cSpFFTPlan1{T<:SpFFTComplex,K} <: SpFFTPlan1{T,K}
  X::Matrix{T}
  F::FFTPlan{T,K}
  w::Vector{T}
  col::Vector{Int}
  p::Vector{Int}
end

immutable rSpFFTPlan1{T<:SpFFTReal,K} <: SpFFTPlan1{T,K}
  X::Matrix{T}
  Xc::Matrix{Complex{T}}
  F::FFTPlan{T,K}
  w::Vector{Complex{T}}
  col::Vector{Int}
  p::Vector{Int}
end

const NB = 128  # default block size; actual can be set on FFT execution

function spfft_blkdiv(n::Integer, k::Integer)
  l = k
  while n % l > 0
    l -= 1
  end
  m = div(n, l)
  l, m
end

function spfft_chkidx{T<:Integer}(n::Integer, idx::AbstractVector{T})
  n > 0 || throw(ArgumentError("n"))
  for i in idx
    (1 <= i <= n) || throw(DomainError())
  end
end

spfft_size(P::SpFFTPlan1) = (size(P.X)..., length(P.w))

# complex transforms

spfft_rc{T<:Real   }(::Type{T}, x) = real(x)  # no inexact error check
spfft_rc{T<:Complex}(::Type{T}, x) =      x

for (f, K) in ((:fft, FORWARD), (:bfft, BACKWARD))
  sf  = Symbol("sp", f)
  psf = Symbol("plan_", sf)
  pf  = Symbol("plan_", f, "!")
  f2s = Symbol(sf, "_f2s!")
  s2f = Symbol(sf, "_s2f!")
  @eval begin
    function $psf{T<:SpFFTComplex,Ti<:Integer}(
        ::Type{T}, n::Integer, idx::AbstractVector{Ti}; args...)
      n > 0 || throw(ArgumentError("n"))
      spfft_chkidx(n, idx)
      k = length(idx)
      l, m = spfft_blkdiv(n, k)
      X = Array(T, l, m)
      F = $pf(X, 1; args...)
      wm = exp(2im*$K*pi/m)
      wn = exp(2im*$K*pi/n)
      w = Array(T, k)
      col = Array(Int, k)
      @inbounds for i = 1:k
        rowm1  = fld(idx[i]-1, l)
        col[i] = idx[i] - rowm1*l
        w[i] = wm^rowm1 * wn^(col[i]-1)
      end
      cSpFFTPlan1{T,$K}(X, F, w, col, sortperm(col))
    end

    function $f2s{T,K,Ty}(
        y::AbstractVector{Ty}, P::cSpFFTPlan1{T,K}, x::AbstractVector{T};
        nb::Integer=$NB)
      l, m, k = spfft_size(P)
      length(x) == l*m || throw(DimensionMismatch)
      length(y) == k   || throw(DimensionMismatch)
      transpose!(P.X, reshape(x,(m,l)))
      P.F*P.X
      y[:] = 0
      nb = min(nb, k)
      s = Array(T, nb)
      idx = 0
      while idx < k
        s[:] = 1
        nbi = min(nb, k-idx)
        @inbounds for j = 1:m
          for i = 1:nbi
            ii = idx + i
            p = P.p[ii]
            z = spfft_rc(Ty, s[i]*P.X[P.col[p],j])
            y[ii] += z
            s[i] *= P.w[p]
          end
        end
        idx += nbi
      end
      ipermute!(y, P.p)
    end

    function $s2f{T,K,Tx}(
        y::AbstractVector{T}, P::cSpFFTPlan1{T,K}, x::AbstractVector{Tx};
        nb::Integer=$NB)
      l, m, k = spfft_size(P)
      length(x) == k   || throw(DimensionMismatch)
      length(y) == l*m || throw(DimensionMismatch)
      P.X[:] = 0
      nb = min(nb, k)
      s = Array(T, nb)
      idx = 0
      xp = x[P.p]
      while idx < k
        s[:] = 1
        nbi = min(nb, k-idx)
        @inbounds for j = 1:m
          for i = 1:nbi
            ii = idx + i
            p = P.p[ii]
            P.X[P.col[p],j] += s[i]*xp[ii]
            s[i] *= P.w[p]
          end
        end
        idx += nbi
      end
      P.F*P.X
      transpose!(reshape(y,(m,l)), P.X)
      y
    end
  end
end

# real transforms
# - only (r2c, f2s) can be further optimized using rfft
# - (r2c, s2f) and (c2r, f2s) basically the same as in c2c case
# - (c2r, s2f) cannot be done in general

function plan_sprfft{T<:SpFFTReal,Ti<:Integer}(
    ::Type{T}, n::Integer, idx::AbstractVector{Ti}; args...)
  n > 0 || throw(ArgumentError("n"))
  spfft_chkidx(n, idx)
  k = length(idx)
  l, m = spfft_blkdiv(n, k)
  Tc = Complex{T}
  X  = Array(T, l, m)
  Xc = Array(Tc, div(l,2)+1, m)
  F = plan_rfft(X, 1; args...)
  wm = exp(-2im*pi/m)
  wn = exp(-2im*pi/n)
  w = Array(Tc, k)
  col = Array(Int, k)
  @inbounds for i = 1:k
    rowm1  = fld(idx[i]-1, l)
    col[i] = idx[i] - rowm1*l
    w[i] = wm^rowm1 * wn^(col[i]-1)
  end
  rSpFFTPlan1{T,FORWARD}(X, Xc, F, w, col, sortperm(col))
end

function sprfft_fullcomplex{T<:SpFFTComplex}(
    X::Matrix{T}, i::Int, j::Int, n::Int, inyq::Int)
  if i <= inyq  return      X[  i  ,j]
  else          return conj(X[n-i+2,j])  # hermitian conjugate
  end
end

function sprfft_f2s!{T,K,Ty<:Complex}(
    y::AbstractVector{Ty}, P::rSpFFTPlan1{T,K}, x::AbstractVector{T};
    nb::Integer=NB)
  l, m, k = spfft_size(P)
  length(x) == l*m || throw(DimensionMismatch)
  length(y) == k   || throw(DimensionMismatch)
  transpose!(P.X, reshape(x,(m,l)))
  A_mul_B!(P.Xc, P.F, P.X)
  y[:] = 0
  nb = min(nb, k)
  s = Array(Complex{T}, nb)
  nyq = size(P.Xc, 1)
  idx = 0
  while idx < k
    s[:] = 1
    nbi = min(nb, k-idx)
    @inbounds for j = 1:m
      for i = 1:nbi
        ii = idx + i
        p = P.p[ii]
        col = P.col[p]
        z = sprfft_fullcomplex(P.Xc, P.col[p], j, l, nyq)
        y[ii] += s[i]*z
        s[i] *= P.w[p]
      end
    end
    idx += nbi
  end
  ipermute!(y, P.p)
end
