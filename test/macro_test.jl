using Test
using Random
using Statistics


"""
Macro-built equivalents of the hand-built models in `models.jl`, restricted to
a single random-walk variable (`x`, not `x{k}` families — dynamic variable
names are deferred, see `rewrites.jl` docstring), for comparing `@model`
output against the hand-built `Sequence`/`Loop` trees it should reproduce.
"""
@model function random_walk1(T::Int)
    x ~ Normal(0, 1)
    for t in 1:T
        x ~ Normal(x, 1)
    end
end

@model function ssm_filter(data, a, q, r)
    x ~ Normal(0, 1)
    for y in data
        x ~ Normal(a * x, q)
        y => Normal(x, r)
    end
end

@model function ssm_filter_weight(data, a, q, r)
    x ~ Normal(0, 1)
    for y in data
        x ~ Normal(a * x, q)
        _ ~ Normal(x, r, y)
    end
end

"""
Same as `ssm_filter_resampled` below, but with an `if resampled ... end`
block (`Cond`) whose body is a no-op `x .= x`, to confirm `Cond`/`if` don't
change the model's distribution and correctly plumb `state.resampled`
through.
"""
@model function ssm_filter_cond(data, a, q, r)
    x ~ Normal(0, 1)
    for y in data
        x ~ Normal(a * x, q)
        y => Normal(x, r)
        if resampled
            x .= x
        end
    end
end

@model function ssm_filter_resampled(data, a, q, r)
    x ~ Normal(0, 1)
    for y in data
        x ~ Normal(a * x, q)
        y => Normal(x, r)
    end
end

const normal_kernels = (Normal=NormalKernel,)

"""
    random_walk1_macro_correctness_test(; T, N, atol)

Matches `random_walk_correctness_test` (K=1 case): `x(T) ~ Normal(0, sqrt(T+1))`.
"""
function random_walk1_macro_correctness_test(; T=10, N=100_000, atol=0.15)
    Random.seed!(42)

    model = random_walk1(T; kernels=normal_kernels)
    state = SMCState(ColumnStore(N))
    apply!(model, state)

    xs = getcol(state.store, :x)
    true_var = T + 1
    return isapprox(mean(xs), 0.0, atol=atol) && isapprox(var(xs), true_var, rtol=0.05)
end

@testset "Macro: random walk (K=1)" begin
    @test random_walk1_macro_correctness_test()
end

function generate_ssm_data(T, a, q, r)
    x_prev = randn()
    data = Float64[]
    for _ in 1:T
        x_curr = a * x_prev + q * randn()
        push!(data, x_curr + r * randn())
        x_prev = x_curr
    end
    return data
end

function ssm_macro_correctness_test(build_fn; T=5, N=200_000, a=0.8, q=0.5, r=0.5, max_abs_diff=0.5, mean_atol=0.3)
    Random.seed!(42)
    data = generate_ssm_data(T, a, q, r)
    exact_mean, exact_evidence = kalman_filter_evidence(data, a, q, r)

    model = build_fn(data, a, q, r; kernels=normal_kernels)
    state = SMCState(ColumnStore(N))
    apply!(model, state)

    weights = state.weights
    evidence = logsumexp(weights) - log(N)
    w = exp.(weights .- maximum(weights))
    w ./= sum(w)
    est_mean = sum(w .* getcol(state.store, :x))

    return isapprox(evidence, exact_evidence, atol=max_abs_diff) && isapprox(est_mean, exact_mean, atol=mean_atol)
end

@testset "Macro: Observe against exact Kalman filter" begin
    @test ssm_macro_correctness_test(ssm_filter)
end

@testset "Macro: Weight against exact Kalman filter" begin
    @test ssm_macro_correctness_test(ssm_filter_weight)
end

function ssm_resampled_macro_correctness_test(build_fn=ssm_filter_resampled; T=50, N=10_000, a=0.8, q=0.5, r=0.5, ess_perc_min=0.5, max_abs_diff=3.0, mean_atol=1.0)
    Random.seed!(42)
    data = generate_ssm_data(T, a, q, r)
    exact_mean, exact_evidence = kalman_filter_evidence(data, a, q, r)

    model = build_fn(data, a, q, r; kernels=normal_kernels)
    state = SMCState(ColumnStore(N); ess_perc_min=ess_perc_min)
    apply!(model, state)

    weights = state.weights
    evidence = logsumexp(weights) - log(N)
    w = exp.(weights .- maximum(weights))
    w ./= sum(w)
    est_mean = sum(w .* getcol(state.store, :x))

    return isapprox(evidence, exact_evidence, atol=max_abs_diff) && isapprox(est_mean, exact_mean, atol=mean_atol)
end

@testset "Macro: Resample against exact Kalman filter (T=50)" begin
    @test ssm_resampled_macro_correctness_test()
end

@testset "Macro: Cond/if (no-op body) matches Resample-only model" begin
    @test ssm_resampled_macro_correctness_test(ssm_filter_cond)
end

"""
    cond_unit_test()

Direct (non-macro) unit test of `Cond`: builds `Cond(predfn, body)` by hand
and checks `apply!` runs `body` iff `predfn(state)` is `true`.
"""
function cond_unit_test()
    state = SMCState(ColumnStore(10))
    broadcast_setcol!(state.store, :x, identity, (Ref(0.0),))

    body = Assign(:x, _ -> Ref(1.0))

    apply!(Cond(_ -> false, body), state)
    all_zero = all(==(0.0), getcol(state.store, :x))

    apply!(Cond(_ -> true, body), state)
    all_one = all(==(1.0), getcol(state.store, :x))

    return all_zero && all_one
end

@testset "Cond transformer (direct)" begin
    @test cond_unit_test()
end
