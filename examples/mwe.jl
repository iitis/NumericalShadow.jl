using Base.Threads
using LinearAlgebra
using BenchmarkTools

function no_bug(samples::Int)
    Threads.@threads for i=1:samples
        g = randn(ComplexF64, 3, 3)
    end
end

function bug(samples::Int)
    Threads.@threads for i=1:samples
        g = randn(ComplexF64, 3, 3)
        q, r = qr(g)
    end
end

BLAS.set_num_threads(1)
samples = 1_000_000

bug(10)
@btime bug(samples)
no_bug(10)
@btime no_bug(samples)