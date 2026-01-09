using NumericalShadow
using BenchmarkTools
using KernelAbstractions
using LinearAlgebra
using Printf
using Plots

# Try to load CUDA
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

using NumericalShadow
using BenchmarkTools
using KernelAbstractions
using LinearAlgebra
using Printf
using Plots

# Try to load CUDA
const HAS_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

function run_benchmarks()
    # Parameters
    ds = [2, 3, 4, 5, 8, 16, 32, 64, 100]
    batch_sizes = [1000, 5000, 10000, 20000]
    T = ComplexF64
    
    # Functions to benchmark
    functions = [
        ("random_unitary", (backend, T_val, d_val, batch_val) -> NumericalShadow.random_unitary(backend, T_val, d_val, batch_val)),
        ("random_pure", (backend, T_val, d_val, batch_val) -> NumericalShadow.random_pure(backend, T_val, d_val, batch_val))
    ]
    
    # Plotting setup
    gr() 
    
    for (func_name, func) in functions
        println("\nBenchmarking $func_name...")
        
        # Store results: dim x batch
        current_results_cpu = zeros(length(ds), length(batch_sizes))
        current_results_cuda = zeros(length(ds), length(batch_sizes))
        
        for (i, d) in enumerate(ds)
            for (j, batch) in enumerate(batch_sizes)
                @printf("%s: d=%d, batch=%d\n", func_name, d, batch)
                
                # CPU
                b_cpu = @benchmark $func(CPU(), $T, $d, $batch) seconds=1.0
                current_results_cpu[i, j] = median(b_cpu).time / 1e6 # ms
                
                if HAS_CUDA
                    # CUDA Warmup
                    func(CUDABackend(), T, d, batch)
                    CUDA.synchronize()
                    
                    b_cuda = @benchmark (CUDA.@sync $func(CUDABackend(), $T, $d, $batch)) seconds=1.0
                    current_results_cuda[i, j] = median(b_cuda).time / 1e6 # ms
                end
            end
        end
        
        # --- Plotting for this function ---
        
        # 1. Dimension Scaling (fixed batch=10000)
        p_dim = plot(title="$func_name Performance (batch=10000)", xlabel="Dimension (d)", ylabel="Time (ms)", legend=:topleft, yaxis=:log10)
        idx_b = findfirst(==(10000), batch_sizes)
        plot!(p_dim, ds, current_results_cpu[:, idx_b], label="CPU", marker=:circle)
        if HAS_CUDA
            plot!(p_dim, ds, current_results_cuda[:, idx_b], label="CUDA", marker=:square)
        end
        savefig(p_dim, joinpath("benchmarks", "benchmark_$(func_name)_d_scaling.png"))
        
        # 2. Batch Scaling for small d (2, 3, 5, 8, 32) explicitly requested or interesting
        plot_ds = [2, 3, 5, 32]
        for d_val in plot_ds
            if d_val in ds
                idx_d = findfirst(==(d_val), ds)
                p_batch = plot(title="$func_name Performance (d=$d_val)", xlabel="Batch Size", ylabel="Time (ms)", legend=:topleft, yaxis=:log10)
                plot!(p_batch, batch_sizes, current_results_cpu[idx_d, :], label="CPU", marker=:circle)
                if HAS_CUDA
                    plot!(p_batch, batch_sizes, current_results_cuda[idx_d, :], label="CUDA", marker=:square)
                end
                savefig(p_batch, joinpath("benchmarks", "benchmark_$(func_name)_batch_scaling_d$(d_val).png"))
            end
        end

        # 3. Overall Speedup (CPU/CUDA)
        if HAS_CUDA
            speedup = current_results_cpu ./ current_results_cuda
            p_speedup = plot(title="$func_name CUDA Speedup", xlabel="Dimension (d)", ylabel="Speedup (x)", legend=:topleft)
            for (j, batch) in enumerate(batch_sizes)
                plot!(p_speedup, ds, speedup[:, j], label="batch=$batch", marker=:circle)
            end
            savefig(p_speedup, joinpath("benchmarks", "benchmark_$(func_name)_speedup.png"))
        end
    end
    
    println("Benchmarks complete. Plots saved in benchmarks/ directory.")
end

run_benchmarks()
