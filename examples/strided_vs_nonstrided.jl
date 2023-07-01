using CUDA

# compare gelsBatched, geqrfBatched and CPU generation
# the GPU routines will probably require to use inv and multiply the inpu by R^-1 to obtain Q (getri, getrf)

function non_strided(a, b)
    println("start view")
    CUDA.@time x = [@view a[:, :, i] for i in axes(a, 3)]
    CUDA.@time y = [@view b[:, :, i] for i in axes(b, 3)]
    println("end view")
    CUDA.CUBLAS.gemm_batched('N', 'N', 1.0f0, x, y)
end

function strided(a, b)
    CUDA.CUBLAS.gemm_strided_batched('N', 'N', 1.0f0, a, b)
end

function generate_matrices(d, nbatch)
    a = CUDA.rand(d, d, nbatch)
    b = CUDA.rand(d, d, nbatch)
    a, b
end

a, b = generate_matrices(10, 10)
non_strided(a, b)
strided(a, b)

a, b = generate_matrices(4, 10^7)
CUDA.@time CUDA.@sync non_strided(a, b)
CUDA.@time CUDA.@sync strided(a, b)

a, b = generate_matrices(1000, 32)
CUDA.@time CUDA.@sync non_strided(a, b)
CUDA.@time CUDA.@sync strided(a, b)
