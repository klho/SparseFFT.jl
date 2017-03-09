#= src/SparseFFT.jl

References:
  F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert. A fast randomized algorithm for
    the approximation of matrices. Appl. Comput. Harmon. Anal. 25: 335-366,
    2008.
=#

module SparseFFT

export

  # spfft1.jl
  cSpFFTPlan1,
  plan_spfft, plan_spbfft,
  spfft_f2s!, spfft_s2f!

# common

const SpFFTComplex = FFTW.fftwComplex
const SpFFTReal    = FFTW.fftwReal
const FORWARD      = FFTW.FORWARD
const BACKWARD     = FFTW.BACKWARD
const FFTPlan      = FFTW.FFTWPlan

function spfft_blkdiv(n::Integer, k::Integer)
  l = k
  while n % l > 0
    l -= 1
  end
  m = div(n, l)
  l, m
end

# source files

include("spfft1.jl")

end # module
