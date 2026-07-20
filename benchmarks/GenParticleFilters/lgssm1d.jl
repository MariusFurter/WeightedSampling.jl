using Pkg
Pkg.activate(@__DIR__)

using Gen
using GenParticleFilters
using WeightedSampling
using Random
using Printf
using Statistics: mean

#=
Comparing performance and usability of WeightedSampling vs GenParticleFilters
on the 1D linear-Gaussian state-space model (`lgssm1d`), matching the LibBi
reference model in `benchmarks/ssm/libbi/lgssm1d/LGSSM1D.bi`:

    x(0) ~ Normal(0, x0_std)
    x(t) = a * x(t-1) + w(t),   w(t) ~ Normal(0, q)
    y(t) ~ Normal(x(t), r)

Organized into 4 parts (each with its model(s), benchmark function(s), and
comments on the issues/tradeoffs found while writing it):
  (1) WeightedSampling, retaining the FULL particle trajectory.
  (2) A naive (no `Unfold`) Gen model: demonstrates O(T^2*N) scaling in T.
  (3) The same Gen model rewritten with the `Unfold` combinator ("elegant"
      O(T*N) version), benchmarked head-to-head against (1).
  (4) A single incremental step: Gen's `pf_update!` vs. WeightedSampling's
      equivalent single `apply!` call.

`compare_all()` at the bottom runs all four, in order.
=#

# =============================================================================
# Shared utilities
# =============================================================================

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
log_evidence)`, used as a correctness check for every particle filter below.
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

# =============================================================================
# (1) WeightedSampling, retaining the full trajectory
# =============================================================================
#
# ISSUE: Gen retains the ENTIRE particle trace history (every random choice
# ever made, e.g. `:xs => 1 => :x`, `:xs => 1 => :y`, ..., `:xs => T => :x`),
# so both memory and the constant-factor cost of extending/copying traces
# grow with `T` (allocation scales ~linearly with `T`, confirmed below: e.g.
# T=200,N=1000 allocates ~1.7 GiB, T=500,N=1000 allocates ~4.3 GiB).
# WeightedSampling's `ColumnStore`, by contrast, only keeps each particle's
# *current* column values -- it has no built-in "trace" concept, so
# retaining the full history is an explicit modeling choice (and cost), not
# automatic. To make this a fair head-to-head rather than an
# apples-to-oranges comparison, the model below explicitly accumulates the
# whole trajectory into per-timestep dynamic variables `x{1}, x{2}, ...,
# x{T+1}` (cf. `examples/2D_ssm.jl`), matching what Gen does by default.

"""
1D linear-Gaussian SSM in WeightedSampling, retaining the ENTIRE trajectory
as dynamic variables `x{1}, x{2}, ..., x{T+1}` (cf. `examples/2D_ssm.jl`).
"""
@model function lgssm1d_ws_traj(data, a, q, r, x0_std)
    x{1} ~ Normal(0.0, x0_std)
    for (t, y) in enumerate(data)
        x{t + 1} ~ Normal(a * x{t}, q)
        y => Normal(x{t + 1}, r)
    end
end

"""
    run_benchmark_ws_traj(; T=100, N=1000, ...)

Benchmark (1): WeightedSampling, retaining the full `x` trajectory.
"""
function run_benchmark_ws_traj(; T=100, N=1000, a=0.9, q=1.0, r=0.5, x0_std=1.0, seed=42)
    Random.seed!(seed)
    _, data = simulate_lgssm1d(T, a, q, r, x0_std)

    # Warm-up (small T/N) to exclude JIT compilation from the timed run.
    run!(lgssm1d_ws_traj(data[1:2], a, q, r, x0_std), SMCState(100))

    model = lgssm1d_ws_traj(data, a, q, r, x0_std)
    state = SMCState(N)
    stats = @timed run!(model, state)
    elapsed = stats.time - stats.compile_time

    exact_mean, exact_evidence = kalman_filter_evidence(data, a, q, r, x0_std)
    x_final = state[WeightedSampling.dynname(:x, T + 1)]
    lml = log_evidence(state)

    println("=" ^ 60)
    println("(1) WeightedSampling, full trajectory retained")
    println("=" ^ 60)
    @printf("T=%d, N=%d\n", T, N)
    @printf("Elapsed time: %.3f s\n", elapsed)
    @printf("Allocated: %.2f MiB\n", stats.bytes / 2^20)
    @printf("Posterior mean (filter): %.4f, exact: %.4f\n",
        sum(WeightedSampling.exp_norm(state.weights) .* x_final), exact_mean)
    @printf("Log evidence (filter): %.4f, exact: %.4f\n\n", lml, exact_evidence)

    return state
end

# =============================================================================
# (2) Naive Gen model (no Unfold): quadratic scaling in T
# =============================================================================
#
# ISSUE: the built-in dynamic DSL always performs an end-to-end
# re-execution of the generative function body whenever `pf_update!` changes
# the arguments (here, growing `T` by one). Concretely, each call to
# `pf_update!(state, (t, ...), (UnknownChange(), ...), obs)` re-runs the
# `for t in 1:T` loop from scratch for every particle: it revisits all `t-1`
# previously-visited addresses `{t' => :x}`/`{t' => :y}` (reusing their
# already-sampled values, so no new randomness is drawn, but still paying
# the cost of stepping through the Julia code and looking up each choice in
# the trace) before finally sampling/scoring the one new step `t`. Summed
# over all `T` update calls, that's `1 + 2 + ... + T = O(T^2)` total
# per-particle work (times `N` particles), i.e. `O(T^2 * N)` overall -- even
# though only `O(T)` *new* random choices are ever made. See part (3) below
# for a model structured so that Gen can avoid this re-execution.

"""
1D linear-Gaussian state-space model, identical to the LibBi reference model
in `benchmarks/ssm/libbi/lgssm1d/LGSSM1D.bi` and to the `@model`-based
version in `benchmarks/ssm/WeightedSampling/lgssm1d.jl`. Written as a plain
(dynamic-DSL) `@gen` function, following the basic particle-filter pattern
from the GenParticleFilters.jl docs
(https://probcomp.github.io/GenParticleFilters.jl/stable/).
"""
@gen function lgssm1d(T::Int, a, q, r, x0_std)
    x = {:x0} ~ normal(0.0, x0_std)
    for t in 1:T
        x = {t => :x} ~ normal(a * x, q)
        {t => :y} ~ normal(x, r)
    end
    return x
end

"""
    run_particle_filter(data, a, q, r, x0_std, n_particles; ess_threshold=0.5)

Bootstrap particle filter, following the "Implementing a Basic Particle
Filter" pattern from the GenParticleFilters.jl docs: `pf_initialize` on the
first observation, then repeated `pf_resample!`/`pf_update!` for each
subsequent observation. Stratified resampling is used whenever the
(normalized) effective sample size drops below `ess_threshold * n_particles`,
mirroring `WeightedSampling`'s default `ess_perc_min=0.5` policy.
"""
function run_particle_filter(data, a, q, r, x0_std, n_particles; ess_threshold=0.5)
    T = length(data)

    init_obs = Gen.choicemap((1 => :y, data[1]))
    state = GenParticleFilters.pf_initialize(lgssm1d, (1, a, q, r, x0_std), init_obs, n_particles)

    for t in 2:T
        if effective_sample_size(state) < ess_threshold * n_particles
            pf_resample!(state, :stratified)
        end
        obs = Gen.choicemap((t => :y, data[t]))
        pf_update!(state, (t, a, q, r, x0_std), (UnknownChange(), NoChange(), NoChange(), NoChange(), NoChange()), obs)
    end

    return state
end

"""
    demo_naive_quadratic_scaling(; Ts=[50, 100, 200, 400], N=100, ...)

Benchmark (2): times the naive (no `Unfold`) Gen model across increasing `T`
to demonstrate its O(T^2 * N) scaling directly. If runtime were O(T*N),
`time / T` would stay roughly constant; because it's O(T^2*N), `time / T`
instead grows roughly linearly with `T`.
"""
function demo_naive_quadratic_scaling(; Ts=[50, 100, 200, 400], N=100, a=0.9, q=1.0, r=0.5, x0_std=1.0, seed=42)
    println("=" ^ 60)
    println("(2) Naive (no Unfold) Gen model: O(T^2*N) scaling demonstration")
    println("=" ^ 60)
    @printf("%-8s%-14s%-14s%s\n", "T", "time (s)", "time/T", "(grows ~linearly with T => O(T^2) overall)")

    # Warm-up once so JIT compilation doesn't pollute the first row.
    Random.seed!(seed)
    _, warmup_data = simulate_lgssm1d(2, a, q, r, x0_std)
    run_particle_filter(warmup_data, a, q, r, x0_std, 50)

    for T in Ts
        Random.seed!(seed)
        _, data = simulate_lgssm1d(T, a, q, r, x0_std)
        stats = @timed run_particle_filter(data, a, q, r, x0_std, N)
        elapsed = stats.time - stats.compile_time
        @printf("%-8d%-14.4f%-14.6f\n", T, elapsed, elapsed / T)
    end
    println()
end

# =============================================================================
# (3) Unfold-based Gen model: O(T*N), benchmarked against (1)
# =============================================================================
#
# ISSUE / fix: Gen's `Unfold` combinator tells Gen explicitly that each step
# only depends on the previous step's state, so `pf_update!` can extend the
# trace by simulating just the newly added step instead of re-executing the
# whole loop history -- this turns the `O(T^2 * N)` scaling of part (2)
# into `O(T * N)`. Even with `Unfold`, though, this is still roughly an
# order of magnitude slower per (T, N) than the WeightedSampling and LibBi
# benchmarks elsewhere in this repo (which run T=5000, N=10_000 in ~1s):
#
# - Per-`pf_update!`-call overhead can be reduced by batching several
#   timesteps into one call (see `run_particle_filter_unfold_batched`
#   below), amortizing fixed choicemap-construction/trace-update-dispatch
#   costs. Measured at T=500, N=1000 (single-threaded): batch=1 -> 1.80s,
#   batch=10 -> 0.86s, batch=25 -> 0.80s, batch=50 -> 0.79s -- a real but
#   *bounded* (~2.2x) speedup, with diminishing returns past batch~10-25.
#   Batching also has an ACCURACY cost: it only checks the ESS threshold
#   (and thus only resamples) once per batch, so weights can degenerate
#   significantly between checks -- e.g. the log-evidence estimate
#   (exact ~ -776.0) is -776.19 (close) at batch=1 but -1443.15 (far off)
#   at batch=25. So batch size is not a free performance knob: it also
#   coarsens the resampling schedule and can bias/inflate estimator
#   variance.
# - General framework overhead: Gen's generality (arbitrary generative
#   functions, custom proposals/MCMC rejuvenation, involutive MCMC, trace
#   translators, etc.) is implemented via a generic trace/GFI abstraction
#   with dynamic dispatch and `ChoiceMap` merging -- even the `static` DSL's
#   compiled IR still goes through this generic machinery. WeightedSampling
#   is purpose-built for SMC over fixed-shape columnar particle arrays
#   (`ColumnStore`), with no generic trace/choicemap layer to pay for. This
#   is an intentional Gen tradeoff (genericity/composability vs. raw
#   throughput for this narrow use case), not a bug or missing optimization.

"""
Same model as `lgssm1d`, but built with Gen's `Unfold` combinator (see the
"Speeding Up Inference Using the Unfold Combinator" section of the Gen.jl SMC
tutorial).
"""
@gen (static) function lgssm1d_step(t::Int, x_prev::Float64, a::Float64, q::Float64, r::Float64)
    x ~ normal(a * x_prev, q)
    y ~ normal(x, r)
    return x
end

const lgssm1d_chain = Gen.Unfold(lgssm1d_step)

@gen (static) function lgssm1d_unfold(T::Int, a::Float64, q::Float64, r::Float64, x0_std::Float64)
    x0 ~ normal(0.0, x0_std)
    xs ~ lgssm1d_chain(T, x0, a, q, r)
    return xs
end

"""
    run_particle_filter_unfold(data, a, q, r, x0_std, n_particles; ess_threshold=0.5)

Same as `run_particle_filter`, but using the `Unfold`-based `lgssm1d_unfold`
model and its `:xs => t => :y` / `:xs => t => :x` addresses.
"""
function run_particle_filter_unfold(data, a, q, r, x0_std, n_particles; ess_threshold=0.5)
    T = length(data)

    init_obs = Gen.choicemap((:xs => 1 => :y, data[1]))
    state = GenParticleFilters.pf_initialize(lgssm1d_unfold, (1, a, q, r, x0_std), init_obs, n_particles)

    for t in 2:T
        if effective_sample_size(state) < ess_threshold * n_particles
            pf_resample!(state, :stratified)
        end
        obs = Gen.choicemap((:xs => t => :y, data[t]))
        pf_update!(state, (t, a, q, r, x0_std), (UnknownChange(), NoChange(), NoChange(), NoChange(), NoChange()), obs)
    end

    return state
end

"""
    run_particle_filter_unfold_batched(data, a, q, r, x0_std, n_particles; ess_threshold=0.5, batch=10)

Same as `run_particle_filter_unfold`, but each `pf_update!` call advances the
model by `batch` timesteps at once (constraining all `batch` new `:y`
observations in one choicemap) instead of one timestep at a time. See the
batching discussion above for the speed/accuracy tradeoff this introduces.
Resampling is still checked/performed once per batch (i.e. at most every
`batch` timesteps), which trades off resampling granularity for speed.
"""
function run_particle_filter_unfold_batched(data, a, q, r, x0_std, n_particles; ess_threshold=0.5, batch=10)
    T = length(data)

    init_obs = Gen.choicemap((:xs => 1 => :y, data[1]))
    state = GenParticleFilters.pf_initialize(lgssm1d_unfold, (1, a, q, r, x0_std), init_obs, n_particles)

    t = 1
    while t < T
        if effective_sample_size(state) < ess_threshold * n_particles
            pf_resample!(state, :stratified)
        end
        t_new = min(T, t + batch)
        obs = Gen.choicemap()
        for tt in (t + 1):t_new
            obs[:xs => tt => :y] = data[tt]
        end
        pf_update!(state, (t_new, a, q, r, x0_std), (UnknownChange(), NoChange(), NoChange(), NoChange(), NoChange()), obs)
        t = t_new
    end

    return state
end

function run_benchmark(; T=200, N=1_000, a=0.9, q=1.0, r=0.5, x0_std=1.0, seed=42, unfold=false, batch=1)
    # NOTE: the plain dynamic-DSL model re-executes the whole `for` loop on
    # every `pf_update!` call, so runtime scales as O(T^2 * N) rather than
    # O(T * N) -- `T=5000, N=10_000` (the WeightedSampling/LibBi benchmark
    # defaults) is infeasible for the naive (`unfold=false`) version; these
    # much smaller defaults keep it runnable. `batch` (only used when
    # `unfold=true`) groups `batch` timesteps per `pf_update!` call; see the
    # batching discussion above. `batch=1` (the default) reproduces the
    # unbatched behavior.
    Random.seed!(seed)
    _, data = simulate_lgssm1d(T, a, q, r, x0_std)

    filter_fn = if !unfold
        (args...) -> run_particle_filter(args...; ess_threshold=0.5)
    elseif batch == 1
        (args...) -> run_particle_filter_unfold(args...; ess_threshold=0.5)
    else
        (args...) -> run_particle_filter_unfold_batched(args...; ess_threshold=0.5, batch=batch)
    end
    x_addr = unfold ? (:xs => T => :x) : (T => :x)

    # Warm-up run (small T/N) to exclude JIT compilation from the timed run.
    filter_fn(data[1:2], a, q, r, x0_std, 100)

    stats = @timed filter_fn(data, a, q, r, x0_std, N)
    elapsed = stats.time - stats.compile_time
    state = stats.value

    exact_mean, exact_evidence = kalman_filter_evidence(data, a, q, r, x0_std)
    final_x_mean = mean(state, x_addr)
    lml = log_ml_estimate(state)

    @printf("model=%s, T=%d, N=%d, batch=%d\n", unfold ? "unfold" : "naive", T, N, unfold ? batch : 1)
    @printf("Elapsed time: %.3f s\n", elapsed)
    @printf("Allocated: %.2f MiB\n", stats.bytes / 2^20)
    @printf("Posterior mean (filter): %.4f, exact: %.4f\n", final_x_mean, exact_mean)
    @printf("Log evidence (filter): %.4f, exact: %.4f\n", lml, exact_evidence)

    return state
end

"""
    compare_benchmarks(; kwargs...)

Run both the naive and `Unfold`-based particle filters on the same data/seed,
for a direct before/after comparison of the `Unfold` optimization.
"""
function compare_benchmarks(; kwargs...)
    run_benchmark(; unfold=false, kwargs...)
    println()
    run_benchmark(; unfold=true, kwargs...)
    return nothing
end

# =============================================================================
# (4) Single step: pf_update! vs. apply!
# =============================================================================
#
# ISSUE: measuring ONE incremental filter step in isolation (rather than a
# full T-step run) requires advancing both filters to T-1 first (untimed),
# then timing only the final step. For WeightedSampling this is done by
# reusing the SAME `:x` column across two separately-built `@model`s: a
# fresh `@model` call's `particle_vars` tracking is per-call-local, but
# `x ~ Normal(a*x, q)` still resolves the RHS `x` to `getcol(store, :x)`
# regardless (the LHS is registered as a particle var before the RHS is
# vectorized) -- so a later model can cleanly read/mutate a column an
# earlier, unrelated model run already created on the same `state`, with no
# need to hand-build transformer trees. One extra gotcha: the single-step
# model is a DISTINCT generated Julia function from the multi-step model, so
# it needs its OWN JIT warm-up on a throwaway state, or the timed call would
# include first-call compilation overhead.

"""
1D linear-Gaussian SSM in WeightedSampling, NOT retaining the trajectory
(only the current `x`). Used to build the "advance to T-1" prefix state.
"""
@model function lgssm1d_ws(data, a, q, r, x0_std)
    x ~ Normal(0.0, x0_std)
    for y in data
        x ~ Normal(a * x, q)
        y => Normal(x, r)
    end
end

"""
A single transition + observation step, operating on the `:x` column already
present in `state` (from a previous `lgssm1d_ws` run). Used to benchmark ONE
incremental `apply!` step (mutate + observe, each ESS-gated-resampled) in
isolation, comparable to a single `pf_update!` call.
"""
@model function lgssm1d_ws_step(y, a, q, r)
    x ~ Normal(a * x, q)
    y => Normal(x, r)
end

"""
    bench_single_step(; T=200, N=1000, ...)

Benchmark (4): a single incremental particle-filter step -- ESS-gated
resample + mutate + observe -- comparing Gen's `pf_update!` (`Unfold` model)
against WeightedSampling's equivalent single `apply!` call
(`lgssm1d_ws_step`). Both filters are first advanced to T-1 (untimed), then
only the FINAL step (extending to T) is timed.
"""
function bench_single_step(; T=200, N=1000, a=0.9, q=1.0, r=0.5, x0_std=1.0, seed=42, ess_threshold=0.5)
    Random.seed!(seed)
    _, data = simulate_lgssm1d(T, a, q, r, x0_std)

    ### --- Gen (Unfold), advance to T-1, then time ONE more pf_update! ---
    init_obs = Gen.choicemap((:xs => 1 => :y, data[1]))
    gen_state = GenParticleFilters.pf_initialize(lgssm1d_unfold, (1, a, q, r, x0_std), init_obs, N)
    for t in 2:(T - 1)
        if effective_sample_size(gen_state) < ess_threshold * N
            pf_resample!(gen_state, :stratified)
        end
        obs = Gen.choicemap((:xs => t => :y, data[t]))
        pf_update!(gen_state, (t, a, q, r, x0_std),
            (UnknownChange(), NoChange(), NoChange(), NoChange(), NoChange()), obs)
    end
    gen_stats = @timed begin
        if effective_sample_size(gen_state) < ess_threshold * N
            pf_resample!(gen_state, :stratified)
        end
        obs = Gen.choicemap((:xs => T => :y, data[T]))
        pf_update!(gen_state, (T, a, q, r, x0_std),
            (UnknownChange(), NoChange(), NoChange(), NoChange(), NoChange()), obs)
    end

    ### --- WeightedSampling, advance to T-1, then time ONE more apply! ---
    ws_state = SMCState(N)
    run!(lgssm1d_ws(data[1:(T - 1)], a, q, r, x0_std), ws_state)

    # Warm up the (separately-compiled) single-step model on a throwaway
    # state so the timed call below excludes its JIT compilation.
    warm_state = SMCState(10)
    run!(lgssm1d_ws(data[1:1], a, q, r, x0_std), warm_state)
    run!(lgssm1d_ws_step(data[1], a, q, r), warm_state)

    ws_stats = @timed run!(lgssm1d_ws_step(data[T], a, q, r), ws_state)

    println("=" ^ 60)
    println("(4) Single incremental step: pf_update! vs. apply!")
    println("=" ^ 60)
    @printf("T=%d, N=%d\n", T, N)
    @printf("Gen pf_update! (ESS-gated resample + mutate + observe): %.3f ms, %.3f MiB\n",
        1000 * (gen_stats.time - gen_stats.compile_time), gen_stats.bytes / 2^20)
    @printf("WeightedSampling apply! (ESS-gated resample + mutate + ESS-gated resample + observe): %.3f ms, %.3f MiB\n\n",
        1000 * (ws_stats.time - ws_stats.compile_time), ws_stats.bytes / 2^20)

    return gen_stats, ws_stats
end

# =============================================================================
# Run everything
# =============================================================================

"""
    compare_all()

Runs all four benchmarks described in the module-level comment at the top of
this file, in order, using shared defaults.
"""
function compare_all()
    run_benchmark_ws_traj(; T=100, N=1000)

    demo_naive_quadratic_scaling()

    println("=" ^ 60)
    println("(3) Unfold-based Gen model (elegant rewrite), T=100, N=1000")
    println("=" ^ 60)
    run_benchmark(; unfold=true, T=100, N=1000)
    println()

    bench_single_step(; T=200, N=1000)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_all()
end
