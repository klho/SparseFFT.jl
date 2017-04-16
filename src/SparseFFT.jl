#= src/SparseFFT.jl

References:
  F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert. A fast randomized algorithm for
    the approximation of matrices. Appl. Comput. Harmon. Anal. 25: 335-366,
    2008.
=#

module SparseFFT

using Base.FFTW: fftwReal, fftwComplex, fftwNumber, FORWARD, BACKWARD, FFTWPlan

export
  cSpFFTPlan1, rSpFFTPlan1, brSpFFTPlan1,
  cSpFFTPlan2, rSpFFTPlan2, brSpFFTPlan2,
  plan_spfft, spfft_f2s!, spfft_s2f!,
  plan_spbfft, spbfft_f2s!, spbfft_s2f!,
  plan_sprfft, sprfft_f2s!,
  plan_spbrfft, spbrfft_s2f!

# common

const SpFFTReal    = fftwReal
const SpFFTComplex = fftwComplex
const SpFFTNumber  = fftwNumber
const FFTPlan      = FFTWPlan

# for complex-to-real transpose (copy from master 2421ff2)
const transposebaselength=64
function transpose_f!(f,B::AbstractMatrix,A::AbstractMatrix)
    inds = indices(A)
    indices(B,1) == inds[2] && indices(B,2) == inds[1] || throw(DimensionMismatch(string(f)))

    m, n = length(inds[1]), length(inds[2])
    if m*n<=4*transposebaselength
        @inbounds begin
            for j = inds[2]
                for i = inds[1]
                    B[j,i] = f(A[i,j])
                end
            end
        end
    else
        transposeblock!(f,B,A,m,n,first(inds[1])-1,first(inds[2])-1)
    end
    return B
end
function transposeblock!(f,B::AbstractMatrix,A::AbstractMatrix,m::Int,n::Int,offseti::Int,offsetj::Int)
    if m*n<=transposebaselength
        @inbounds begin
            for j = offsetj+(1:n)
                for i = offseti+(1:m)
                    B[j,i] = f(A[i,j])
                end
            end
        end
    elseif m>n
        newm=m>>1
        transposeblock!(f,B,A,newm,n,offseti,offsetj)
        transposeblock!(f,B,A,m-newm,n,offseti+newm,offsetj)
    else
        newn=n>>1
        transposeblock!(f,B,A,m,newn,offseti,offsetj)
        transposeblock!(f,B,A,m,n-newn,offseti,offsetj+newn)
    end
    return B
end

# source files

include("spfft1.jl")
include("spfft2.jl")

end  # module
