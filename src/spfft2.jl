#= src/spfft2.jl
=#

abstract type SpFFTPlan2{T,K} end

# complex transforms

struct cSpFFTPlan2{T<:SpFFTComplex,K} <: SpFFTPlan2{T,K}
  P1::cSpFFTPlan1{T,K}
  P2::cSpFFTPlan1{T,K}
  k1n2::Matrix{T}
  n2k1::Matrix{T}
  k2k1::Matrix{T}
end

for (fcn, K) in ((:fft, FORWARD), (:bfft, BACKWARD))
  sf  = Symbol("sp", fcn)
  psf = Symbol("plan_", sf)
  f2s = Symbol(sf, "_f2s!")
  s2f = Symbol(sf, "_s2f!")
  @eval begin
    function $psf(::Type{T}, n1::Integer, n2::Integer,
                  idx1::AbstractVector{Ti}, idx2::AbstractVector{Ti};
                  args...) where {T<:SpFFTComplex,Ti<:Integer}
      P1 = $psf(T, n1, idx1; args...)
      P2 = $psf(T, n2, idx2; args...)
      k1 = length(idx1)
      k2 = length(idx2)
      k1n2 = Array{T}(undef, k1, n2)
      n2k1 = Array{T}(undef, n2, k1)
      k2k1 = Array{T}(undef, k2, k1)
      cSpFFTPlan2{T,$K}(P1, P2, k1n2, n2k1, k2k1)
    end

    function $f2s(Y::AbstractMatrix{Ty}, P::cSpFFTPlan2{T,K},
                  X::AbstractMatrix{Tx}; args...) where {T,K,Tx,Ty}
      l1, m1, k1 = spfft_size(P.P1)
      l2, m2, k2 = spfft_size(P.P2)
      n1 = l1*m1
      n2 = l2*m2
      size(X) == (n1, n2) || throw(DimensionMismatch)
      size(Y) == (k1, k2) || throw(DimensionMismatch)
      @inbounds for i = 1:n2
        $f2s(view(P.k1n2,:,i), P.P1, view(X,:,i); args...)
      end
      transpose!(P.n2k1, P.k1n2)
      @inbounds for i = 1:k1
        $f2s(view(P.k2k1,:,i), P.P2, view(P.n2k1,:,i); args...)
      end
      transpose_f!(z->spfft_rc(Ty,z), Y, P.k2k1)
    end

    function $s2f(Y::AbstractMatrix{Ty}, P::cSpFFTPlan2{T,K},
                  X::AbstractMatrix{Tx}; args...) where {T,K,Tx,Ty}
      l1, m1, k1 = spfft_size(P.P1)
      l2, m2, k2 = spfft_size(P.P2)
      n1 = l1*m1
      n2 = l2*m2
      size(X) == (k1, k2) || throw(DimensionMismatch)
      size(Y) == (n1, n2) || throw(DimensionMismatch)
      transpose!(P.k2k1, X)
      @inbounds for i = 1:k1
        $s2f(view(P.n2k1,:,i), P.P2, view(P.k2k1,:,i); args...)
      end
      transpose!(P.k1n2, P.n2k1)
      @inbounds for i = 1:n2
        $s2f(view(Y,:,i), P.P1, view(P.k1n2,:,i); args...)
      end
      Y
    end
  end
end

# real transforms

## (r2c, f2s)

struct rSpFFTPlan2{T<:SpFFTReal,Tc<:SpFFTComplex} <: SpFFTPlan2{T,FORWARD}
  P1::rSpFFTPlan1{T}
  P2::cSpFFTPlan1{Tc,FORWARD}
  k1n2::Matrix{Tc}
  n2k1::Matrix{Tc}
  k2k1::Matrix{Tc}
end

function plan_sprfft(::Type{T}, n1::Integer, n2::Integer,
                     idx1::AbstractVector{Ti}, idx2::AbstractVector{Ti};
                     args...) where {T<:SpFFTReal,Ti<:Integer}
  Tc = Complex{T}
  P1 = plan_sprfft(T, n1, idx1; args...)
  P2 = plan_spfft(Tc, n2, idx2; args...)
  k1 = length(idx1)
  k2 = length(idx2)
  k1n2 = Array{Tc}(undef, k1, n2)
  n2k1 = Array{Tc}(undef, n2, k1)
  k2k1 = Array{Tc}(undef, k2, k1)
  rSpFFTPlan2{T,Tc}(P1, P2, k1n2, n2k1, k2k1)
end

function sprfft_f2s!(
    Y::AbstractMatrix{Ty}, P::rSpFFTPlan2{T}, X::AbstractMatrix{Tx};
    args...) where {T,Tx<:Real,Ty<:Complex}
  l1, m1, k1 = spfft_size(P.P1)
  l2, m2, k2 = spfft_size(P.P2)
  n1 = l1*m1
  n2 = l2*m2
  size(X) == (n1, n2) || throw(DimensionMismatch)
  size(Y) == (k1, k2) || throw(DimensionMismatch)
  @inbounds for i = 1:n2
    sprfft_f2s!(view(P.k1n2,:,i), P.P1, view(X,:,i); args...)
  end
  transpose!(P.n2k1, P.k1n2)
  @inbounds for i = 1:k1
    spfft_f2s!(view(P.k2k1,:,i), P.P2, view(P.n2k1,:,i); args...)
  end
  transpose!(Y, P.k2k1)
end

## (c2r, s2f)
## - input must contain only nonredundant frequencies, i.e., up to index
##   n1/2 + 1 in the first dimension for a full size of n1

struct brSpFFTPlan2{T<:SpFFTReal,Tc<:SpFFTComplex} <: SpFFTPlan2{T,BACKWARD}
  P1::brSpFFTPlan1{T,Tc}
  P2::cSpFFTPlan1{Tc,BACKWARD}
  k1n2::Matrix{Tc}
  n2k1::Matrix{Tc}
  k2k1::Matrix{Tc}
end

function plan_spbrfft(::Type{T}, n1::Integer, n2::Integer,
                      idx1::AbstractVector{Ti}, idx2::AbstractVector{Ti};
                      args...) where {T<:SpFFTReal,Ti<:Integer}
  Tc = Complex{T}
  P1 = plan_spbrfft(T, n1, idx1; args...)
  P2 = plan_spbfft(Tc, n2, idx2; args...)
  k1 = length(idx1)
  k2 = length(idx2)
  k1n2 = Array{T }(undef, k1, n2)
  n2k1 = Array{Tc}(undef, n2, k1)
  k2k1 = Array{Tc}(undef, k2, k1)
  brSpFFTPlan2{T,Tc}(P1, P2, k1n2, n2k1, k2k1)
end

function spbrfft_s2f!(
    Y::AbstractMatrix{Ty}, P::brSpFFTPlan2{T}, X::AbstractMatrix{Tx};
    args...) where {T,Tx<:Complex,Ty<:Real}
  l1, m1, kc1, k1 = spbrfft_size(P.P1)
  l2, m2, k2 = spfft_size(P.P2)
  n1 = l1*m1
  n2 = l2*m2
  size(X) == (k1, k2) || throw(DimensionMismatch)
  size(Y) == (n1, n2) || throw(DimensionMismatch)
  transpose!(P.k2k1, X)
  @inbounds for i = 1:k1
    spbfft_s2f!(view(P.n2k1,:,i), P.P2, view(P.k2k1,:,i); args...)
  end
  transpose!(P.k1n2, P.n2k1)
  @inbounds for i = 1:n2
    spbrfft_s2f!(view(Y,:,i), P.P1, view(P.k1n2,:,i); args...)
  end
  Y
end
