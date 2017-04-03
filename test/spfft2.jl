#= test/spfft1.jl
=#

println("spfft2.jl")
tic()

n1 = 50
n2 = 50
k1 = 10
k2 = 10

# complex transforms

r1 = randperm(n1)[1:k1]
r2 = randperm(n2)[1:k2]

for fcn in (:fft, :bfft)
  sf = Symbol("sp", fcn)
  for T in (Complex64, Complex128)
    println("  $sf/$T")
    Tr = real(T)

    psf = Symbol("plan_", sf)
    @eval P = $psf($T, n1, n2, r1, r2)

    x  = Array(T, n1, n2)
    y  = Array(T, k1, k2)
    xr = Array(Tr, size(x))
    yr = Array(Tr, size(y))

    # f2s
    f2s = Symbol(sf, "_f2s!")
    rand!(x)

    ## (c2c, f2s)
    @eval f = $fcn($x)[r1,r2]
    @eval $f2s($y, P, $x)
    @test_approx_eq f y

    ## (c2r, f2s)
    @eval $f2s($yr, P, $x)
    @test_approx_eq real(f) yr

    ## (r2c, f2s)
    xr[:] = real(x)
    @eval f = $fcn($xr)[r1,r2]
    @eval $f2s($y, P, $xr)
    @test_approx_eq f y

    # s2f
    s2f = Symbol(sf, "_s2f!")
    rand!(y)

    ## (c2c, s2f)
    x[:] = 0
    x[r1,r2] = y
    @eval f = $fcn($x)
    @eval $s2f($x, P, $y)
    @test_approx_eq f x

    ## (c2r, s2f)
    @eval $s2f($xr, P, $y)
    @test_approx_eq real(f) xr

    ## (r2c, s2f)
    yr[:] = real(y)
    x[:] = 0
    x[r1,r2] = yr
    @eval f = $fcn($x)
    @eval $s2f($x, P, $yr)
    @test_approx_eq f x
  end
end

# real transforms

m1 = div(n1,2) + 1
r2 = randperm(n2)[1:k2]

for T in (Float32, Float64)
  Tc = Complex{T}

  x  = Array(T , n1, n2)
  y  = Array(Tc, k1, k2)
  xc = Array(Tc, m1, n2)

  # (r2c, f2s)
  println("  sprfft/$T")
  r1 = randperm(m1)[1:k1]
  P = plan_sprfft(T, n1, n2, r1, r2)
  rand!(x)
  f = rfft(x)[r1,r2]
  sprfft_f2s!(y, P, x)
  @test_approx_eq f y

  # (c2r, s2f)
  println("  spbrfft/$T")
  r1 = vcat([1,m1], 1+randperm(m1-2)[1:k1-2])  # include edge cases
  P = plan_spbrfft(T, n1, n2, r1, r2)
  rand!(y)
  idx1 = (r1 .== 1) | (r1 .== m1)
  y[idx1,:] = real(y[idx1,:])
  xc[:] = 0
  xc[r1,r2] = y
  f = brfft(xc, n1)
  spbrfft_s2f!(x, P, y)
  @test_approx_eq f x
end

toc()
