using Pkg
Pkg.activate(@__DIR__)

using Turing
using AdvancedPS
using Random
using Printf
using Statistics: median
using BenchmarkTools

include(joinpath(@__DIR__, "..", "ssm", "simulate.jl"))

# Suppress Turing's own progress bar; sampling progress is not comparable
# across frameworks and clutters the benchmark output.
Turing.setprogress!(false)

"""
1D linear-Gaussian state-space model, identical to the reference model in
`benchmarks/ssm/libbi/lgssm1d/LGSSM1D.bi` and to the `@model`-based
WeightedSampling implementation in `benchmarks/ssm/WeightedSampling/lgssm1d.jl`:

    x(0) ~ Normal(0, x0_std)
    x(t) = a * x(t-1) + w(t),   w(t) ~ Normal(0, q)
    y(t) ~ Normal(x(t), r)

Built as a Turing.jl `@model`, run with `Turing.Inference.SMC` (the bootstrap
particle filter). Number of particles == the `N` argument passed to `sample`.
"""
Turing.@model function lgssm1d_turing(data, a, q, r, x0_std)
    T = length(data)
    x ~ Normal(0.0, x0_std)
    for t in 2:T
        x ~ Normal(a * x, q)
        data[t] ~ Normal(x, r)
    end
    return x
end

function run_benchmark(; T=50, N=100, a=0.9, q=1.0, r=0.5, x0_std=1.0, seed=42,
    ess_perc_min=1.0)
    Random.seed!(seed)
    _, data = simulate_lgssm1d(T, a, q, r, x0_std)

    # `SMC([resampler, ]threshold)`: threshold=1.0 forces resampling at EVERY
    # step (relative ESS is always <= 1), matching the other frameworks'
    # `ess_perc_min=1.0`/`essThreshold=1.0`/`ESS_REL=1.0` convention in this
    # benchmark suite.
    spl = SMC(AdvancedPS.resample_systematic, ess_perc_min)

    # Warm-up run (small T/N) to exclude JIT compilation from the timed run.
    warmup_model = lgssm1d_turing(data[1:2], a, q, r, x0_std)
    sample(warmup_model, spl, 100; check_model=false)

    model = lgssm1d_turing(data, a, q, r, x0_std)

    stats = @timed sample(model, spl, N; check_model=false)
    chn = stats.value
    elapsed = stats.time - stats.compile_time
    alloc_mib = stats.bytes / 2^20

    exact_mean, exact_evidence = kalman_filter_evidence(data, a, q, r, x0_std)

    w = vec(chn[:weight])
    # `x` is now a scalar per particle (the final filtered state) rather than a
    # trajectory vector, so no per-particle `[end]` indexing is needed.
    xT = vec(chn[:x])
    post_mean = sum(w .* xT)
    log_evidence = chn[:logevidence][1]

    @printf("T=%d, N=%d\n", T, N)
    @printf("Elapsed time: %.3f s\n", elapsed)
    @printf("Allocated: %.2f MiB\n", alloc_mib)
    @printf("Posterior mean (filter): %.4f, exact: %.4f\n", post_mean, exact_mean)
    @printf("Log evidence (filter): %.4f, exact: %.4f\n", log_evidence, exact_evidence)
    @printf("RESULT,Turing,T=%d,N=%d,elapsed_s=%.6f,alloc_mib=%.4f,post_mean=%.6f,exact_mean=%.6f\n",
        T, N, elapsed, alloc_mib, post_mean, exact_mean)

    return chn
end

"""
    bench_single_update(; N=1000, base_T=50, delta=10, ...)

Isolates the cost of ONE mutate+observe+resample update at `N` particles.

Unlike GenParticleFilters' `pf_update!`/WeightedSampling's `apply!`, Turing's
`SMC` has no exposed API for stepping an existing particle-filter state
forward incrementally -- `sample(model, SMC(...), N)` always runs a whole
model (i.e. all `T` steps) in one call. So instead of timing a single
incremental step directly, this uses the same MARGINAL-COST subtraction
methodology as `benchmarks/ssm/SequentialMonteCarlo/lGModel.jl`'s
`bench_single_update.jl` comparison (see repo notes): time a full run at
`base_T` steps vs. `base_T + delta` steps (same data prefix), and divide the
difference by `delta` to isolate the per-step cost while averaging over
`delta` marginal steps (reduces subtraction noise vs. a raw 1-step diff).
The same subtraction is applied to `@benchmark`'s `.memory` (bytes
allocated) field to isolate a per-step allocation estimate.
"""
function bench_single_update(; N=1000, base_T=50, delta=10, a=0.9, q=1.0, r=0.5,
    x0_std=1.0, seed=42, ess_perc_min=1.0)
    Random.seed!(seed)
    _, data = simulate_lgssm1d(base_T + delta, a, q, r, x0_std)
    spl = SMC(AdvancedPS.resample_systematic, ess_perc_min)

    # Warm-up run (small T/N) to exclude JIT compilation from the timed runs.
    warmup_model = lgssm1d_turing(data[1:2], a, q, r, x0_std)
    sample(warmup_model, spl, 100; check_model=false)

    base_model = lgssm1d_turing(data[1:base_T], a, q, r, x0_std)
    extended_model = lgssm1d_turing(data[1:(base_T + delta)], a, q, r, x0_std)

    bench_base = @benchmark(sample($base_model, $spl, $N; check_model=false))
    bench_extended = @benchmark(sample($extended_model, $spl, $N; check_model=false))

    t_base = median(bench_base.times) / 1e9
    t_extended = median(bench_extended.times) / 1e9
    per_step_us = 1e6 * (t_extended - t_base) / delta

    # Same marginal-cost subtraction applied to allocations (`.memory`, in
    # bytes) since there's no incremental API to isolate a single step's
    # allocation directly (unlike Gen's `pf_update!`, benchmarked in-place).
    alloc_per_step_kib = (bench_extended.memory - bench_base.memory) / delta / 2^10

    @printf("N=%d, base_T=%d, delta=%d\n", N, base_T, delta)
    @printf("Per-step median: %.4f us, %.4f KiB (base=%.4fs/%.2f MiB, extended=%.4fs/%.2f MiB)\n",
        per_step_us, alloc_per_step_kib, t_base, bench_base.memory / 2^20,
        t_extended, bench_extended.memory / 2^20)
    # Same framework label as benchmarks/ssm/bench_single_update.jl and
    # benchmarks/GenParticleFilters/lgssm1d.jl so all merge into one CSV via
    # parse_results.py; a `turing_` metric prefix avoids colliding with
    # those scripts' `ws_`/`smc_`/`gen_` metrics.
    @printf("RESULT,bench_single_update,N=%d,turing_median_us=%.4f,turing_alloc_kib=%.4f\n",
        N, per_step_us, alloc_per_step_kib)

    return per_step_us, alloc_per_step_kib
end

# =============================================================================
# CLI entry point
# =============================================================================
#
# Usage:
#   julia lgssm1d.jl                 # full run, T=5000, N=10_000 (module defaults)
#   julia lgssm1d.jl full [T] [N]    # run_benchmark, RESULT line
#   julia lgssm1d.jl single [N]      # bench_single_update, RESULT line
function main(args)
    if isempty(args)
        run_benchmark()
        return nothing
    end

    mode = args[1]
    if mode == "full"
        T = length(args) >= 2 ? parse(Int, args[2]) : 5000
        N = length(args) >= 3 ? parse(Int, args[3]) : 10_000
        run_benchmark(; T=T, N=N)
    elseif mode == "single"
        N = length(args) >= 2 ? parse(Int, args[2]) : 1000
        bench_single_update(; N=N)
    else
        error("Unknown mode \"$mode\"; expected \"full\" or \"single\".")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
