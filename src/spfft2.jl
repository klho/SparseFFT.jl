#= src/spfft2.jl
=#

immutable cSpFFTPlan2{T<:SpFFTComplex,K}
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
    function $psf{T<:SpFFTComplex,Ti<:Integer}(
        ::Type{T}, n1::Integer, n2::Integer,
        idx1::AbstractVector{Ti}, idx2::AbstractVector{Ti}; args...)
      P1 = $psf(T, n1, idx1; args...)
      P2 = $psf(T, n2, idx2; args...)
      k1 = length(idx1)
      k2 = length(idx2)
      k1n2 = Array(T, k1, n2)
      n2k1 = Array(T, n2, k1)
      k2k1 = Array(T, k2, k1)
      cSpFFTPlan2{T,$K}(P1, P2, k1n2, n2k1, k2k1)
    end

    function $f2s{T<:SpFFTComplex,K}(
        Y::AbstractMatrix{T}, P::cSpFFTPlan2{T,K}, X::AbstractMatrix{T};
        args...)
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
      transpose!(Y, P.k2k1)
    end

    function $s2f{T<:SpFFTComplex,K}(
        Y::AbstractMatrix{T}, P::cSpFFTPlan2{T,K}, X::AbstractMatrix{T};
        args...)
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
