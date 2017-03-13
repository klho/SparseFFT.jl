#= test/spfft1.jl
=#

println("spfft2.jl")
tic()

n1 = 50
n2 = 50
k1 = 10
k2 = 10

r1 = randperm(n1)[1:k1]
r2 = randperm(n2)[1:k2]

for fcn in (:fft, :bfft)
  sf = Symbol("sp", fcn)
  for T in (Complex64, Complex128)
    println("  $sf/$T")

    psf = Symbol("plan_sp", fcn)
    @eval P = $psf($T, n1, n2, r1, r2)

    x = rand(T, n1, n2)
    y = rand(T, k1, k2)

    @eval f = $fcn($x)[r1,r2]
    spfft_f2s!(y, P, x)
    @test_approx_eq f y

    x[:] = 0
    x[r1,r2] = y
    @eval f = $fcn($x)
    spfft_s2f!(x, P, y)
    @test_approx_eq f x
  end
end

toc()
