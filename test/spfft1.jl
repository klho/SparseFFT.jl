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

    psf = Symbol("plan_sp", fcn)
    @eval P = $psf($T, n, r)

    x = rand(T, n)
    y = rand(T, k)

    @eval f = $fcn($x)[r]
    spfft_f2s!(y, P, x)
    @test_approx_eq f y

    x[:] = 0
    x[r] = y
    @eval f = $fcn($x)
    spfft_s2f!(x, P, y)
    @test_approx_eq f x
  end
end

toc()
