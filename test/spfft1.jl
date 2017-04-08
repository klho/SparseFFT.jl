#= test/spfft1.jl
=#

println("spfft1.jl")
tic()

n = 1000
k =  101

# complex transforms

r = randperm(n)[1:k]

for fcn in (:fft, :bfft)
  sf = Symbol("sp", fcn)
  for T in (Complex64, Complex128)
    println("  $sf/$T")
    Tr = real(T)

    psf = Symbol("plan_", sf)
    @eval P = $psf($T, n, r)

    x  = Array(T, n)
    y  = Array(T, k)
    xr = Array(Tr, size(x))
    yr = Array(Tr, size(y))

    # f2s
    f2s = Symbol(sf, "_f2s!")
    rand!(x)

    ## (c2c, f2s)
    @eval f = $fcn($x)[r]
    @eval $f2s($y, P, $x)
    @test_approx_eq f y

    ## (c2r, f2s)
    @eval $f2s($yr, P, $x)
    @test_approx_eq real(f) yr

    ## (r2c, f2s)
    xr[:] = real(x)
    @eval f = $fcn($xr)[r]
    @eval $f2s($y, P, $xr)
    @test_approx_eq f y

    # s2f
    s2f = Symbol(sf, "_s2f!")
    rand!(y)

    ## (c2c, s2f)
    x[:] = 0
    x[r] = y
    @eval f = $fcn($x)
    @eval $s2f($x, P, $y)
    @test_approx_eq f x

    ## (c2r, s2f)
    @eval $s2f($xr, P, $y)
    @test_approx_eq real(f) xr

    ## (r2c, s2f)
    yr[:] = real(y)
    x[:] = 0
    x[r] = yr
    @eval f = $fcn($x)
    @eval $s2f($x, P, $yr)
    @test_approx_eq f x
  end
end

# real transforms

m = div(n,2) + 1

for T in (Float32, Float64)
  Tc = Complex{T}

  x  = Array(T , n)
  y  = Array(Tc, k)
  xc = Array(Tc, m)

  # (r2c, f2s)
  println("  sprfft/$T")
  r = randperm(m)[1:k]
  P = plan_sprfft(T, n, r)
  rand!(x)
  f = rfft(x)[r]
  sprfft_f2s!(y, P, x)
  @test_approx_eq f y

  # (c2r, s2f)
  println("  spbrfft/$T")
  r = vcat([1,m], 1+randperm(m-2)[1:k-2])  # include edge cases
  P = plan_spbrfft(T, n, r)
  rand!(y)
  idx = (r .== 1) | (r .== m)
  y[idx] = real(y[idx])
  xc[:] = 0
  xc[r] = y
  f = brfft(xc, n)
  spbrfft_s2f!(x, P, y)
  @test_approx_eq f x
end

toc()
