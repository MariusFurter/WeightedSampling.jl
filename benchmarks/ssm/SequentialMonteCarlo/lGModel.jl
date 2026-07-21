using Pkg
Pkg.activate(@__DIR__)

using SequentialMonteCarlo
using RNGPool
using Random
using Printf

include(joinpath(@__DIR__, "..", "simulate.jl"))

"""
1D linear-Gaussian state-space model, identical in structure/parameters to
the LibBi reference model in `benchmarks/ssm/libbi/lgssm1d/LGSSM1D.bi` and to
the `@model`-based implementation in
`benchmarks/ssm/WeightedSampling/lgssm1d.jl`:

    x(0) ~ Normal(0, x0_std)
    x(t) = a * x(t-1) + w(t),   w(t) ~ Normal(0, q)
    y(t) ~ Normal(x(t), r)

Built directly against SequentialMonteCarlo.jl's low-level `SMCModel` API
(gold-standard baseline), following the exact style of `SMCExamples.jl`'s
`lgModel.jl`/`lgKalman.jl` (immutable `M!`/`lG` closures with precomputed
constants, `Float64Particle` from that package's `particles.jl`).
"""

mutable struct Float64Particle
    x::Float64
    Float64Particle() = new()
end

@inline function Base.:(==)(x::Float64Particle, y::Float64Particle)
    return x.x == y.x
end

"""
    make_lgssm1d_model(ys, a, q, r, x0_std)

Build an `SMCModel` for the model above, observing the data `ys` (length `n`,
so `maxn = n`, `p == 1` is the initial step `x(0) ~ Normal(0, x0_std)`, and
`p > 1` are the transition+observation steps).
"""
function make_lgssm1d_model(ys::Vector{Float64}, a, q, r, x0_std)
    n::Int64 = length(ys)
    logncG::Float64 = -0.5 * log(2 * π * r^2)
    invRover2::Float64 = 0.5 / r^2

    @inline function lG(p::Int64, particle::Float64Particle, ::Nothing)
        @inbounds v::Float64 = particle.x - ys[p]
        return logncG - v * invRover2 * v
    end

    @inline function M!(newParticle::Float64Particle, rng::RNG, p::Int64,
        particle::Float64Particle, ::Nothing)
        if p == 1
            newParticle.x = x0_std * randn(rng)
        else
            newParticle.x = a * particle.x + q * randn(rng)
        end
    end

    return SMCModel(M!, lG, n, Float64Particle, Nothing)
end

function run_benchmark(; n=5000, N=10_000, a=0.9, q=1.0, r=0.5, x0_std=1.0, seed=42,
    ess_perc_min=1.0)
    if Threads.nthreads() != 1
        error("Threads.nthreads()=$(Threads.nthreads()); this script requires " *
              "Threads.nthreads()==1. RNGPool's engine pool is populated at " *
              "process start (module `__init__`) via `Threads.@threads`, and " *
              "with >1 threads some engine slots can be left `#undef`, causing " *
              "`getRNG()` to throw `UndefRefError` — this can NOT be fixed by " *
              "changing anything at runtime (`Threads.nthreads()` is fixed for " *
              "the life of the Julia process). Restart Julia with `-t 1` (e.g. " *
              "`julia -t 1` from an external terminal, or set the VS Code Julia " *
              "extension's `julia.NumThreads` setting to 1 and restart the REPL).")
    end

    setRNGs(0)
    Random.seed!(seed)
    _, data = simulate_lgssm1d(n, a, q, r, x0_std)

    model = make_lgssm1d_model(data, a, q, r, x0_std)

    # SMCIO's `essThreshold` gates resampling the opposite way round from
    # WeightedSampling's `ess_perc_min`: `smc!` SKIPS resampling at step `p`
    # when `esses[p] > essThreshold` (see SequentialMonteCarlo.jl's
    # `serial.jl`), i.e. it resamples whenever the relative ESS falls to (or
    # below) `essThreshold` — exactly mirroring `ess_perc_min` (WeightedSampling
    # resamples whenever `ess_perc < ess_perc_min`), so passing `ess_perc_min`
    # straight through as `essThreshold` reproduces the same resampling rule.
    # `ess_perc_min=1.0` (the default here, matching WeightedSampling's
    # `SMCState` default) means "always resample", since relative ESS is
    # always <= 1.
    essThreshold = ess_perc_min

    # Warm-up run (small n/N) to exclude JIT compilation from the timed run.
    warmup_n = min(2, n)
    warmup_model = make_lgssm1d_model(data[1:warmup_n], a, q, r, x0_std)
    warmup_smcio = SMCIO{Float64Particle, Nothing}(100, warmup_n, 1, false, essThreshold)
    smc!(warmup_model, warmup_smcio)

    smcio = SMCIO{Float64Particle, Nothing}(N, n, 1, false, essThreshold)

    stats = @timed smc!(model, smcio)
    elapsed = stats.time - stats.compile_time
    alloc_mib = stats.bytes / 2^20

    exact_mean, exact_evidence = kalman_filter_evidence(data, a, q, r, x0_std)
    filter_mean = SequentialMonteCarlo.eta(smcio, p -> p.x, true, n)
    filter_evidence = smcio.logZhats[n]

    @printf("n=%d, N=%d\n", n, N)
    @printf("Elapsed time: %.3f s\n", elapsed)
    @printf("Allocated: %.2f MiB\n", alloc_mib)
    @printf("Posterior mean (filter): %.4f, exact: %.4f\n", filter_mean, exact_mean)
    @printf("Log evidence (filter): %.4f, exact: %.4f\n", filter_evidence, exact_evidence)
    @printf("RESULT,SequentialMonteCarlo,T=%d,N=%d,elapsed_s=%.6f,alloc_mib=%.4f,post_mean=%.6f,exact_mean=%.6f\n",
        n, N, elapsed, alloc_mib, filter_mean, exact_mean)

    return smcio
end

if abspath(PROGRAM_FILE) == @__FILE__
    T = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5000
    N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10_000
    run_benchmark(; n=T, N=N)
end
