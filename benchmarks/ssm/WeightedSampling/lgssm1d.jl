using WeightedSampling
using Random
using Printf

include(joinpath(@__DIR__, "..", "simulate.jl"))

"""
1D linear-Gaussian state-space model, identical to the LibBi reference model
in `benchmarks/ssm/libbi/lgssm1d/LGSSM1D.bi`:

    x(0) ~ Normal(0, x0_std)
    x(t) = a * x(t-1) + w(t),   w(t) ~ Normal(0, q)
    y(t) ~ Normal(x(t), r)

Built with the `@model` macro (no explicit `kernels=` kwarg needed; `Normal`
is resolved via `WeightedSampling.default_kernels`).
"""
@model function lgssm1d(data, a, q, r, x0_std)
    x ~ Normal(0.0, x0_std)
    for y in data
        x ~ Normal(a * x, q)
        y => Normal(x, r)
    end
end

function run_benchmark(; T=5000, N=10_000, a=0.9, q=1.0, r=0.5, x0_std=1.0, seed=42,
    ess_perc_min=1.0)
    if Threads.nthreads() != 1
        @warn "Running with Threads.nthreads()=$(Threads.nthreads()); " *
              "launch Julia with `-t 1` (or `JULIA_NUM_THREADS=1`) to match " *
              "the single-threaded (`--nthreads 1`) LibBi benchmark for a fair comparison."
    end

    Random.seed!(seed)
    _, data = simulate_lgssm1d(T, a, q, r, x0_std)

    # Warm-up run (small T/N) to exclude JIT compilation from the timed run.
    warmup_model = lgssm1d(data[1:2], a, q, r, x0_std)
    warmup_state = SMCState(100; ess_perc_min=ess_perc_min)
    run!(warmup_model, warmup_state)

    model = lgssm1d(data, a, q, r, x0_std)
    state = SMCState(N; ess_perc_min=ess_perc_min)

    stats = @timed run!(model, state)
    elapsed = stats.time - stats.compile_time
    alloc_mib = stats.bytes / 2^20

    exact_mean, exact_evidence = kalman_filter_evidence(data, a, q, r, x0_std)
    xs = state[:x]
    post_mean = sum(WeightedSampling.exp_norm(state.weights) .* xs)

    @printf("T=%d, N=%d\n", T, N)
    @printf("Elapsed time: %.3f s\n", elapsed)
    @printf("Allocated: %.2f MiB\n", alloc_mib)
    @printf("Posterior mean (filter): %.4f, exact: %.4f\n", post_mean, exact_mean)
    @printf("RESULT,WeightedSampling,T=%d,N=%d,elapsed_s=%.6f,alloc_mib=%.4f,post_mean=%.6f,exact_mean=%.6f\n",
        T, N, elapsed, alloc_mib, post_mean, exact_mean)

    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    T = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5000
    N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10_000
    run_benchmark(; T=T, N=N)
end
