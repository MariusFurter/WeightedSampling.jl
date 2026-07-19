using Test
using Random
using Statistics


"""
    random_walk_correctness_test(; K=4, T=10, N=100_000, atol=0.1)

Sanity-check the hand-built `Sequence` random walk against the known analytic
marginal: `x_k(T) ~ Normal(0, sqrt(T + 1))` (variance 1 from the initial
sample, plus 1 per step). Checks empirical mean/variance across all `K`
independent walks (pooled for a tighter check at fixed `N`).
"""
function random_walk_correctness_test(; K=4, T=10, N=100_000, atol=0.15, make_store=ColumnStore)
    Random.seed!(42)

    model = build_random_walk(K, T)
    state = SMCState(make_store(N))

    apply!(model, state)

    final_cols = [getcol(state.store, Symbol(:x, k)) for k in 1:K]
    pooled = reduce(vcat, final_cols)

    true_var = T + 1
    mean_ok = isapprox(mean(pooled), 0.0, atol=atol)
    var_ok = isapprox(var(pooled), true_var, rtol=0.05)

    return mean_ok && var_ok
end

"""
    random_walk_loop_correctness_test(; K, T, N, atol)

Same check as `random_walk_correctness_test`, but for `build_random_walk_loop`
(uses the `Loop` transformer for the `T`-step update loop instead of
unrolling it), to confirm `Loop`/`apply!` give the same distribution.
"""
function random_walk_loop_correctness_test(; K=4, T=10, N=100_000, atol=0.15, make_store=ColumnStore)
    Random.seed!(42)

    model = build_random_walk_loop(K, T)
    state = SMCState(make_store(N))

    apply!(model, state)

    final_cols = [getcol(state.store, Symbol(:x, k)) for k in 1:K]
    pooled = reduce(vcat, final_cols)

    true_var = T + 1
    mean_ok = isapprox(mean(pooled), 0.0, atol=atol)
    var_ok = isapprox(var(pooled), true_var, rtol=0.05)

    return mean_ok && var_ok
end

@testset "Hand-built random walk Sequence" begin
    @test random_walk_correctness_test()
end

@testset "Loop-based random walk Sequence" begin
    @test random_walk_loop_correctness_test()
end

### Kalman filter log-evidence test for `Observe`
# Linear-Gaussian SSM. This model does not resample, so it is plain importance
# sampling over the whole sequence -- keep T small to avoid weight degeneracy.

"""
    ssm_observe_correctness_test(; T, N, a, q, r, max_abs_diff)

Compares the `Observe`-based `build_ssm_filter` against the exact Kalman
filter solution: log-evidence (`logsumexp(weights) - log(N)`) and posterior
mean of the final state.
"""
function ssm_observe_correctness_test(; T=5, N=200_000, a=0.8, q=0.5, r=0.5, max_abs_diff=0.5, mean_atol=0.3, make_store=ColumnStore)
    Random.seed!(42)

    x_prev = randn()
    data = Float64[]
    for _ in 1:T
        x_curr = a * x_prev + q * randn()
        push!(data, x_curr + r * randn())
        x_prev = x_curr
    end

    exact_mean, exact_evidence = kalman_filter_evidence(data, a, q, r)

    model = build_ssm_filter(data, a, q, r)
    state = SMCState(make_store(N))
    apply!(model, state)

    weights = state.weights
    evidence = logsumexp(weights) - log(N)

    w = exp.(weights .- maximum(weights))
    w ./= sum(w)
    est_mean = sum(w .* getcol(state.store, :x))

    evidence_ok = isapprox(evidence, exact_evidence, atol=max_abs_diff)
    mean_ok = isapprox(est_mean, exact_mean, atol=mean_atol)

    return evidence_ok && mean_ok
end

@testset "Hand-built: Observe against exact Kalman filter" begin
    @test ssm_observe_correctness_test()
end

"""
    ssm_weight_correctness_test(; T, N, a, q, r, max_abs_diff, mean_atol)

Same check as `ssm_observe_correctness_test`, but for `build_ssm_filter_weight`
(uses `Weight` instead of `Observe`), to confirm they agree.
"""
function ssm_weight_correctness_test(; T=5, N=200_000, a=0.8, q=0.5, r=0.5, max_abs_diff=0.5, mean_atol=0.3, make_store=ColumnStore)
    Random.seed!(42)

    x_prev = randn()
    data = Float64[]
    for _ in 1:T
        x_curr = a * x_prev + q * randn()
        push!(data, x_curr + r * randn())
        x_prev = x_curr
    end

    exact_mean, exact_evidence = kalman_filter_evidence(data, a, q, r)

    model = build_ssm_filter_weight(data, a, q, r)
    state = SMCState(make_store(N))
    apply!(model, state)

    weights = state.weights
    evidence = logsumexp(weights) - log(N)

    w = exp.(weights .- maximum(weights))
    w ./= sum(w)
    est_mean = sum(w .* getcol(state.store, :x))

    evidence_ok = isapprox(evidence, exact_evidence, atol=max_abs_diff)
    mean_ok = isapprox(est_mean, exact_mean, atol=mean_atol)

    return evidence_ok && mean_ok
end

@testset "Hand-built: Weight against exact Kalman filter" begin
    @test ssm_weight_correctness_test()
end

"""
    ssm_resampled_correctness_test(; T, N, a, q, r, ess_perc_min, max_abs_diff, mean_atol)

Same check as `ssm_observe_correctness_test`, but for
`build_ssm_filter_resampled` (resamples every step via `Resample`), over a
longer horizon `T` that would otherwise degenerate without resampling —
mirrors `test/kalman_evidence_test.jl`'s T=50 case.
"""
function ssm_resampled_correctness_test(; T=50, N=10_000, a=0.8, q=0.5, r=0.5, ess_perc_min=0.5, max_abs_diff=3.0, mean_atol=1.0, make_store=ColumnStore)
    Random.seed!(42)

    x_prev = randn()
    data = Float64[]
    for _ in 1:T
        x_curr = a * x_prev + q * randn()
        push!(data, x_curr + r * randn())
        x_prev = x_curr
    end

    exact_mean, exact_evidence = kalman_filter_evidence(data, a, q, r)

    model = build_ssm_filter_resampled(data, a, q, r)
    state = SMCState(make_store(N); ess_perc_min=ess_perc_min)
    apply!(model, state)

    weights = state.weights
    evidence = logsumexp(weights) - log(N)

    w = exp.(weights .- maximum(weights))
    w ./= sum(w)
    est_mean = sum(w .* getcol(state.store, :x))

    evidence_ok = isapprox(evidence, exact_evidence, atol=max_abs_diff)
    mean_ok = isapprox(est_mean, exact_mean, atol=mean_atol)

    return evidence_ok && mean_ok
end

@testset "Hand-built: Resample against exact Kalman filter (T=50)" begin
    @test ssm_resampled_correctness_test()
end
