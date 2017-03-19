#= src/SparseFFT.jl

References:
  F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert. A fast randomized algorithm for
    the approximation of matrices. Appl. Comput. Harmon. Anal. 25: 335-366,
    2008.
=#

module SparseFFT

using Base.FFTW: fftwComplex, fftwReal, FORWARD, BACKWARD, FFTWPlan

export

  # spfft1.jl
  cSpFFTPlan1, rSpFFTPlan1,
  plan_spfft, spfft_f2s!, spfft_s2f!,
  plan_spbfft, spbfft_f2s!, spbfft_s2f!,
  plan_sprfft, sprfft_f2s!

# common

const SpFFTComplex = fftwComplex
const SpFFTReal    = fftwReal
const FFTPlan      = FFTWPlan

# source files

include("spfft1.jl")
include("spfft2.jl")

end # module
