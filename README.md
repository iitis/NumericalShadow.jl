# NumericalShadow.jl

Numerical tools for estimating numerical shadows and q-shadows, with random-state
and random-unitary generators for CPU and CUDA backends.

## Installation

```julia
using Pkg
Pkg.add("NumericalShadow")
```

For local development:

```julia
using Pkg
Pkg.develop(path=".")
```

## Quick Start (CPU)

```julia
using NumericalShadow
using KernelAbstractions

A = ComplexF64[1 0; 0 -1]
samples = 10_000

sampler(backend, d, n) = random_pure(backend, ComplexF64, d, n)
h = shadow(CPU(), A, samples, sampler, 2_000)
```

## Quick Start (CUDA, Optional)

```julia
using NumericalShadow
using KernelAbstractions
using CUDA

if CUDA.functional()
    backend = CUDABackend()
    U = random_unitary(backend, ComplexF64, 8, 1024)
    @show size(U)  # (8, 8, 1024)
end
```

## Benchmarks

Run the dedicated CUDA QR benchmark:

```bash
julia --project=benchmarks benchmarks/benchmark_cuda_batched_qr.jl
```

Run the broader benchmark suite:

```bash
julia --project=benchmarks benchmarks/runbenchmarks.jl
```

## Example Plots (Shadow.pdf Problems 2/3/4)

Generate sample datasets:

```bash
julia --project=. examples/problem2_k_shadow.jl
julia --project=. examples/problem3_c_shadow.jl
julia --project=. examples/problem4_C_shadow.jl
```

These examples now write HDF5 files (`.h5`) to `examples/results/`.

Render PNG plots (Julia + CairoMakie):

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/plot_problem_shadows.jl
```

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Enable slower statistical tests:

```bash
NUMERICALSHADOW_RUN_SLOW_TESTS=true julia --project=. -e 'using Pkg; Pkg.test()'
```

## Public API

- `move_to_backend`
- `shadow`, `qshadow`, `product_qshadow`
- `random_pure`, `random_overlap`, `random_unitary`, `random_max_ent`
