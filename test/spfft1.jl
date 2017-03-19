#= test/spfft1.jl
=#

println("spfft1.jl")
tic()

n = 1000
k =  100

r = randperm(n)[1:k]

for fcn in (:fft, :bfft)
  sf = Symbol("sp", fcn)
  for T in (Complex64, Complex128)
    println("  $sf/$T")

    psf = Symbol("plan_", sf)
    @eval P = $psf($T, n, r)

    x =  rand(T, n)
    y = Array(T, k)

    # (c2c, f2s)
    @eval f = $fcn($x)[r]
    f2s = Symbol(sf, "_f2s!")
    @eval $f2s($y, P, $x)
    @test_approx_eq f y

    # (c2r, f2s)
    z = Array(real(T), k)
    @eval $f2s($z, P, $x)
    @test_approx_eq real(f) z

    # (c2c, s2f)
    rand!(y)
    x[:] = 0
    x[r] = y
    @eval f = $fcn($x)
    s2f = Symbol(sf, "_s2f!")
    @eval $s2f($x, P, $y)
    @test_approx_eq f x

    # (r2c, s2f)
    z = real(y)
    x[:] = 0
    x[r] = z
    @eval f = $fcn($x)
    @eval $s2f($x, P, $z)
    @test_approx_eq f x
  end
end

# (r2c, f2s)
for T in (Float32, Float64)
  println("  sprfft/$T")
  Tc = Complex{T}

  P = plan_sprfft(T, n, r)

  x = rand(T , n)
  y = rand(Tc, k)

  f = fft(x)[r]
  sprfft_f2s!(y, P, x)
  @test_approx_eq f y
end

toc()
