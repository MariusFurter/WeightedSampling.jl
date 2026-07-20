using WeightedSampling
using Random
using Printf

"""
1D linear-Gaussian state-space model, identical to the LibBi reference model
in `benchmarks/linear_gaussian/libbi/lgssm1d/LGSSM1D.bi`:

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

"""
    simulate_lgssm1d(T, a, q, r, x0_std)

Simulate one trajectory/observation sequence from the model above, returning
`(states, observations)` of length `T`.
"""
function simulate_lgssm1d(T::Int, a, q, r, x0_std)
    states = Vector{Float64}(undef, T)
    observations = Vector{Float64}(undef, T)
    x = x0_std * randn()
    for t in 1:T
        x = a * x + q * randn()
        states[t] = x
        observations[t] = x + r * randn()
    end
    return states, observations
end

"""
    kalman_filter_evidence(data, a, q, r, x0_std)

Exact Kalman filter for this model. Returns `(posterior_mean_final,
log_evidence)`, used as a correctness check for the particle filter.
"""
function kalman_filter_evidence(data, a, q, r, x0_std)
    μ, P = 0.0, x0_std^2
    log_evidence = 0.0
    for y in data
        μ_pred = a * μ
        P_pred = a^2 * P + q^2

        S = P_pred + r^2
        residual = y - μ_pred
        log_evidence += -0.5 * (log(2π) + log(S) + residual^2 / S)

        K = P_pred / S
        μ = μ_pred + K * residual
        P = (1 - K) * P_pred
    end
    return μ, log_evidence
end

function run_benchmark(; T=5000, N=10_000, a=0.9, q=1.0, r=0.5, x0_std=1.0, seed=42)
    if Threads.nthreads() != 1
        @warn "Running with Threads.nthreads()=$(Threads.nthreads()); " *
              "launch Julia with `-t 1` (or `JULIA_NUM_THREADS=1`) to match " *
              "the single-threaded (`--nthreads 1`) LibBi benchmark for a fair comparison."
    end

    Random.seed!(seed)
    _, data = simulate_lgssm1d(T, a, q, r, x0_std)

    # Warm-up run (small T/N) to exclude JIT compilation from the timed run.
    warmup_model = lgssm1d(data[1:2], a, q, r, x0_std)
    warmup_state = SMCState(100)
    run!(warmup_model, warmup_state)

    model = lgssm1d(data, a, q, r, x0_std)
    state = SMCState(N)

    stats = @timed run!(model, state)
    elapsed = stats.time - stats.compile_time

    exact_mean, exact_evidence = kalman_filter_evidence(data, a, q, r, x0_std)
    xs = state[:x]

    @printf("T=%d, N=%d\n", T, N)
    @printf("Elapsed time: %.3f s\n", elapsed)
    @printf("Allocated: %.2f MiB\n", stats.bytes / 2^20)
    @printf("Posterior mean (filter): %.4f, exact: %.4f\n", sum(WeightedSampling.exp_norm(state.weights) .* xs), exact_mean)

    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
