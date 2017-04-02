#= src/spfft1.jl
=#

abstract SpFFTPlan1{T,K}

const NB = 128  # default block size; actual used can be set on FFT execution

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
# - underlying FFT is c2c but can also take real input/output
# -- if real  input, will convert from real to complex
# -- if real output, will take real part
# - note: (r2c, f2s) and (c2r, s2f) can be further optimized

immutable cSpFFTPlan1{T<:SpFFTComplex,K} <: SpFFTPlan1{T,K}
  X::Matrix{T}
  F::FFTPlan{T,K}
  w::Vector{T}
  col::Vector{Int}
  p::Vector{Int}
end

spfft_rc{T<:Real   }(::Type{T}, x) = real(x)  # no inexact error check
spfft_rc{T<:Complex}(::Type{T}, x) =      x

function spfft_rc!{T,Tx}(::Type{T}, x::AbstractArray{Tx})
  @simd for i in 1:length(x)
    @inbounds x[i] = spfft_rc(T, x[i])
  end
  x
end

for (f, K) in ((:fft, FORWARD), (:bfft, BACKWARD))
  sf  = Symbol("sp", f)
  psf = Symbol("plan_", sf)
  pf  = Symbol("plan_", f, "!")
  f2s = Symbol(sf, "_f2s!")
  s2f = Symbol(sf, "_s2f!")
  @eval begin
    function $psf{T<:SpFFTComplex,Ti<:Integer}(
        ::Type{T}, n::Integer, idx::AbstractVector{Ti}; args...)
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
        rowm1, colm1 = divrem(idx[i]-1, l)
        w[i] = wm^rowm1 * wn^colm1
        col[i] = colm1 + 1
      end
      p = sortperm(col)
      cSpFFTPlan1{T,$K}(X, F, w[p], col[p], p)
    end

    function $f2s{T,K,Tx,Ty}(
        y::AbstractVector{Ty}, P::cSpFFTPlan1{T,K}, x::AbstractVector{Tx};
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
            z = spfft_rc(Ty, s[i]*P.X[P.col[ii],j])
            y[ii] += z
            s[i] *= P.w[ii]
          end
        end
        idx += nbi
      end
      ipermute!(y, P.p)
    end

    function $s2f{T,K,Tx,Ty}(
        y::AbstractVector{Ty}, P::cSpFFTPlan1{T,K}, x::AbstractVector{Tx};
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
            P.X[P.col[ii],j] += s[i]*xp[ii]
            s[i] *= P.w[ii]
          end
        end
        idx += nbi
      end
      P.F*P.X
      transpose_f!(_->spfft_rc(Ty,_), reshape(y,(m,l)), P.X)
    end
  end
end

# real transforms

## (r2c, f2s)

immutable rSpFFTPlan1{T<:SpFFTReal} <: SpFFTPlan1{T,FORWARD}
  X::Matrix{T}
  Xc::Matrix{Complex{T}}
  F::FFTPlan{T,FORWARD}
  w::Vector{Complex{T}}
  col::Vector{Int}
  p::Vector{Int}
end

function sprfft_fullcomplex{T<:SpFFTComplex}(
    X::Matrix{T}, i::Integer, j::Integer, n::Integer, inyq::Integer)
  if i <= inyq  return      X[  i  ,j]
  else          return conj(X[n-i+2,j])  # hermitian conjugate
  end
end

function plan_sprfft{T<:SpFFTReal,Ti<:Integer}(
    ::Type{T}, n::Integer, idx::AbstractVector{Ti}; args...)
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
    rowm1, colm1 = divrem(idx[i]-1, l)
    w[i] = wm^rowm1 * wn^colm1
    col[i] = colm1 + 1
  end
  p = sortperm(col)
  rSpFFTPlan1{T}(X, Xc, F, w[p], col[p], p)
end

function sprfft_f2s!{T,Ty<:Complex}(
    y::AbstractVector{Ty}, P::rSpFFTPlan1{T}, x::AbstractVector{T};
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
        z = sprfft_fullcomplex(P.Xc, P.col[ii], j, l, nyq)
        y[ii] += s[i]*z
        s[i] *= P.w[ii]
      end
    end
    idx += nbi
  end
  ipermute!(y, P.p)
end

## (c2r, s2f)
## - input must contain only nonredundant frequencies

immutable brSpFFTPlan1{T<:SpFFTReal,TF<:SpFFTComplex} <: SpFFTPlan1{T,BACKWARD}
  X::Matrix{T}
  Xc::Matrix{Complex{T}}
  F::FFTPlan{TF,BACKWARD}
  k::Int
  w::Vector{Complex{T}}
  col::Vector{Int}
  p::Vector{Int}
end

function spbrfft_fullsize{T<:Integer}(idx::AbstractVector{T}, n::Integer)
  d, r = divrem(n, 2)
  nyq = r == 0 ? d+1 : 0
  n = 0
  for i in idx
    n += (i == 1 || i == nyq) ? 1 : 2
  end
  n
end

spbrfft_size(P::brSpFFTPlan1) = (spfft_size(P)..., P.k)

function plan_spbrfft{T<:SpFFTReal,Ti<:Integer}(
    ::Type{T}, n::Integer, idx::AbstractVector{Ti}; args...)
  nyqm1 = div(n, 2)
  spfft_chkidx(nyqm1+1, idx)
  k = length(idx)
  kf = spbrfft_fullsize(idx, n)
  l, m = spfft_blkdiv(n, k)
  Tc = Complex{T}
  X  = Array(T, l, m)
  Xc = Array(Tc, div(l,2)+1, m)
  F = plan_brfft(Xc, l, 1; args...)
  wm = exp(2im*pi/m)
  wn = exp(2im*pi/n)
  w = Array(Tc, kf)
  col = Array(Int, kf)
  xp  = Array(Int, kf)
  a = 1
  b = kf
  for i = 1:k
    idxm1 = idx[i] - 1
    rowm1, colm1 = divrem(idxm1, l)
      w[a] = wm^rowm1 * wn^colm1
    col[a] = colm1 + 1
     xp[a] = i
    a += 1
    (idxm1 == 0 || idxm1 == nyqm1) && continue
    idxm1 = n - idxm1
    rowm1, colm1 = divrem(idxm1, l)
      w[b] = wm^rowm1 * wn^colm1
    col[b] = colm1 + 1
     xp[b] = -i
    b -= 1
  end
  keep = col .<= size(Xc,1)
    w =   w[keep]
  col = col[keep]
   xp =  xp[keep]
  p = sortperm(col)
  brSpFFTPlan1{T,Tc}(X, Xc, F, k, w[p], col[p], xp[p])
end

function spbrfft_s2f!{T,Tx<:Complex}(
    y::AbstractVector{T}, P::brSpFFTPlan1{T}, x::AbstractVector{Tx};
    nb::Integer=NB)
  l, m, kc, k = spbrfft_size(P)
  length(x) == k   || throw(DimensionMismatch)
  length(y) == l*m || throw(DimensionMismatch)
  P.Xc[:] = 0
  nb = min(nb, kc)
  s = Array(Complex{T}, nb)
  idx = 0
  xp = Array(Tx, kc)
  for i = 1:kc
    p = P.p[i]
    xp[i] = p > 0 ? x[p] : conj(x[-p])
  end
  while idx < kc
    s[:] = 1
    nbi = min(nb, kc-idx)
    @inbounds for j = 1:m
      for i = 1:nbi
        ii = idx + i
        P.Xc[P.col[ii],j] += s[i]*xp[ii]
        s[i] *= P.w[ii]
      end
    end
    idx += nbi
  end
  A_mul_B!(P.X, P.F, P.Xc)
  transpose!(reshape(y,(m,l)), P.X)
  y
end
