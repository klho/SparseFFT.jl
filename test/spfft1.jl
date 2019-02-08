#= test/spfft1.jl
=#

println("spfft1.jl")
@time begin

@testset "spfft1" begin

n = 1500
k =  151

# complex transforms

r = randperm(n)[1:k]

for fcn in (:fft, :bfft)
  sf = Symbol("sp", fcn)
  for T in (ComplexF32, ComplexF64)
    println("  $sf/$T")
    Tr = real(T)

    psf = Symbol("plan_", sf)
    @eval P = $psf($T, $n, $r)

    x  = Array{T }(undef, n)
    y  = Array{T }(undef, k)
    xr = Array{Tr}(undef, size(x))
    yr = Array{Tr}(undef, size(y))

    # f2s
    f2s = Symbol(sf, "_f2s!")
    rand!(x)

    ## (c2c, f2s)
    @eval f = $fcn($x)[$r]
    @eval $f2s($y, P, $x)
    @test f ≈ y

    ## (c2r, f2s)
    @eval $f2s($yr, P, $x)
    @test real(f) ≈ yr

    ## (r2c, f2s)
    xr[:] = real(x)
    @eval f = $fcn($xr)[$r]
    @eval $f2s($y, P, $xr)
    @test f ≈ y

    # s2f
    s2f = Symbol(sf, "_s2f!")
    rand!(y)

    ## (c2c, s2f)
    fill!(x, 0)
    x[r] = y
    @eval f = $fcn($x)
    @eval $s2f($x, P, $y)
    @test f ≈ x

    ## (c2r, s2f)
    @eval $s2f($xr, P, $y)
    @test real(f) ≈ xr

    ## (r2c, s2f)
    yr[:] = real(y)
    fill!(x, 0)
    x[r] = yr
    @eval f = $fcn($x)
    @eval $s2f($x, P, $yr)
    @test f ≈ x
  end
end

# real transforms

m = div(n,2) + 1
r = vcat([1,m], 1 .+ randperm(m-2)[1:k-2])  # include edge cases

for T in (Float32, Float64)
  Tc = Complex{T}

  x  = Array{T }(undef, n)
  y  = Array{Tc}(undef, k)
  xc = Array{Tc}(undef, m)

  # (r2c, f2s)
  println("  sprfft/$T")
  P = plan_sprfft(T, n, r)
  rand!(x)
  f = rfft(x)[r]
  sprfft_f2s!(y, P, x)
  @test f ≈ y

  # (c2r, s2f)
  println("  spbrfft/$T")
  P = plan_spbrfft(T, n, r)
  rand!(y)
  y[r.==1] = real(y[r.==1])
  even = n % 2 == 0
  even && (y[r.==m] = real(y[r.==m]))
  fill!(xc, 0)
  xc[r] = y
  f = brfft(xc, n)
  spbrfft_s2f!(x, P, y)
  @test f ≈ x
end

end
end
