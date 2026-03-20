# Shadow.pdf mapping:
#   Problem 1 (convexoid vs normal case for A(d), d in [0, 1/2]).
# This script sweeps d and computes numerical shadow samples for the
# non-normal extension of diag(1, ω, ω^2).

using NumericalShadow
using KernelAbstractions
using LinearAlgebra
using CUDA
using ProgressMeter
using Printf

const DEFAULT_SAMPLES = 1_000_000_000
const DEFAULT_BATCHSIZE = 10_000_000
const DEFAULT_D_START = 0.0
const DEFAULT_D_STOP = 0.5
const DEFAULT_D_STEP = 0.01

env_flag(name::AbstractString, default::Bool) =
    lowercase(get(ENV, name, default ? "true" : "false")) in ("1", "true", "yes")

function choose_backend()
    use_cuda = env_flag("NUMERICALSHADOW_USE_CUDA", true)
    if use_cuda && CUDA.functional()
        println("Using CUDA backend.")
        return CUDABackend()
    end
    println("Using CPU backend.")
    return CPU()
end

function matrix_for_d(d::Real)
    return cat(
        Diagonal([1, exp(2im * pi / 3), exp(4im * pi / 3)]),
        [0 d; 0 0];
        dims = (1, 2),
    )
end

function run_hidden_non_normal_family(
    backend,
    d_values::AbstractVector{<:Real},
    samples::Int,
    batchsize::Int;
    sample_type::Type,
    family_label::AbstractString,
    out_prefix::AbstractString,
    overwrite::Bool,
)
    mkpath(joinpath(@__DIR__, "results"))
    @showprogress 2 "Iterating d ($family_label)" for d in d_values
        d_tag = @sprintf("%.2f", d)
        out_file = joinpath(@__DIR__, "results", "$(out_prefix)$(d_tag).h5")
        if !overwrite && isfile(out_file)
            continue
        end

        A = matrix_for_d(d)
        sampling_f = (b, dd, n) -> NumericalShadow.random_pure(b, sample_type, dd, n)
        shadow = NumericalShadow.shadow(backend, A, samples, sampling_f, batchsize)
        shadow.nr = NumericalShadow.numerical_range(A)
        shadow.evs = eigvals(A)
        NumericalShadow.save(shadow, out_file)
    end
end

function main()
    backend = choose_backend()

    samples = parse(Int, get(ENV, "NUMERICALSHADOW_SAMPLES", string(DEFAULT_SAMPLES)))
    batchsize = parse(Int, get(ENV, "NUMERICALSHADOW_BATCHSIZE", string(DEFAULT_BATCHSIZE)))
    d_start = parse(Float64, get(ENV, "NUMERICALSHADOW_D_START", string(DEFAULT_D_START)))
    d_stop = parse(Float64, get(ENV, "NUMERICALSHADOW_D_STOP", string(DEFAULT_D_STOP)))
    d_step = parse(Float64, get(ENV, "NUMERICALSHADOW_D_STEP", string(DEFAULT_D_STEP)))
    overwrite = env_flag("NUMERICALSHADOW_OVERWRITE", false)

    samples > 0 || throw(ArgumentError("`samples` must be positive, got $samples"))
    batchsize > 0 || throw(ArgumentError("`batchsize` must be positive, got $batchsize"))
    d_step > 0 || throw(ArgumentError("`d_step` must be positive, got $d_step"))
    d_start <= d_stop || throw(ArgumentError("Require `d_start <= d_stop`, got $d_start > $d_stop"))

    d_values = collect(d_start:d_step:d_stop)
    isempty(d_values) && throw(ArgumentError("No d values generated for the configured range"))

    println("Hidden non-normal sweep configuration:")
    println("  samples = $samples")
    println("  batchsize = $batchsize")
    println("  d_start = $d_start, d_stop = $d_stop, d_step = $d_step")
    println("  number of d points = $(length(d_values))")
    println("  overwrite existing files = $overwrite")

    run_hidden_non_normal_family(
        backend,
        d_values,
        samples,
        batchsize;
        sample_type = ComplexF32,
        family_label = "complex vectors",
        out_prefix = "hidden_non_normal_complex_",
        overwrite = overwrite,
    )
    run_hidden_non_normal_family(
        backend,
        d_values,
        samples,
        batchsize;
        sample_type = Float32,
        family_label = "real vectors",
        out_prefix = "hidden_non_normal_real_",
        overwrite = overwrite,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
