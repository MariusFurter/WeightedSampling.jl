using Test
using Random
using Statistics
using Distributions


"""
    move_cancellation_test(; T=3, N=1_000, step_size=0.3)

Verify that adding a target-independent factor to the scored path leaves the
`Move`'s accept decisions unchanged, confirming "score everything,
target-independent factors cancel."

Builds two models sharing the same initial `θ` column and the same `T`
observations `y[t] => Normal(θ, 1)` (which DO depend on the move target
`θ`): model `B` additionally has one extra `Observe` on a fixed value `z`
against `Normal(0, 1)` — a factor that does NOT depend on `θ` at all. Running
an identical `Move` (same RNG seed reset immediately before each `apply!`, so
the proposal draw and the accept-decision draw are identical) on both should
give the same accepted/rejected `θ` values, since the extra factor
contributes identically to `s_old` and `s_new` and cancels in `Δ`.
"""
function move_cancellation_test(; T=3, N=1_000, step_size=0.3)
    Random.seed!(1)
    y = randn(T)
    z = 0.7  # fixed, independent of θ
    θ0 = randn(N)

    store_A = ColumnStore(N)
    state_A = SMCState(store_A)
    broadcast_setcol!(store_A, :θ, identity, (copy(θ0),))
    root_A = Sequence(
        Sample(:θ, NormalKernel, s -> (Ref(0.0), Ref(1.0))),
        (Observe(s -> Ref(y[t]), NormalKernel, s -> (getcol(s.store, :θ), Ref(1.0))) for t in 1:T)...,
    )
    state_A.root = root_A
    state_A.depth = T + 1

    store_B = ColumnStore(N)
    state_B = SMCState(store_B)
    broadcast_setcol!(store_B, :θ, identity, (copy(θ0),))
    root_B = Sequence(
        Sample(:θ, NormalKernel, s -> (Ref(0.0), Ref(1.0))),
        (Observe(s -> Ref(y[t]), NormalKernel, s -> (getcol(s.store, :θ), Ref(1.0))) for t in 1:T)...,
        Observe(s -> Ref(z), NormalKernel, s -> (Ref(0.0), Ref(1.0))),
    )
    state_B.root = root_B
    state_B.depth = T + 2

    move = Move([:θ], RW, s -> (step_size,))

    Random.seed!(2)
    apply!(move, state_A)
    Random.seed!(2)
    apply!(move, state_B)

    return isapprox(getcol(store_A, :θ), getcol(store_B, :θ), atol=1e-9)
end

"""
    move_invariance_test(; T=5, τ0=2.0, σ=1.0, N=200_000, n_moves=20, step_size=0.3)

MH invariance for a static parameter. Model: `θ ~ Normal(0, τ0)`, `T`
observations `y[t] => Normal(θ, σ)`.
Particles are initialized EXACTLY at the closed-form Normal-Normal posterior;
a sweep of `RW` moves on `θ` should leave the (empirical) posterior mean and
variance unchanged within tolerance.
"""
function move_invariance_test(; T=5, τ0=2.0, σ=1.0, N=200_000, n_moves=20, step_size=0.3)
    Random.seed!(42)
    y = randn(T) .* σ .+ 1.3

    prior_var = τ0^2
    obs_var = σ^2
    post_var = 1 / (1 / prior_var + T / obs_var)
    post_mean = post_var * (sum(y) / obs_var)

    store = ColumnStore(N)
    state = SMCState(store)
    broadcast_setcol!(store, :θ, identity, (randn(N) .* sqrt(post_var) .+ post_mean,))

    root = Sequence(
        Sample(:θ, NormalKernel, s -> (Ref(0.0), Ref(τ0))),
        (Observe(s -> Ref(y[t]), NormalKernel, s -> (getcol(s.store, :θ), Ref(σ))) for t in 1:T)...,
    )
    state.root = root
    state.depth = T + 1

    move = Move([:θ], RW, s -> (step_size,))
    for _ in 1:n_moves
        apply!(move, state)
    end

    θ = getcol(store, :θ)
    mean_ok = isapprox(mean(θ), post_mean, atol=0.05)
    var_ok = isapprox(var(θ), post_var, atol=0.05)
    return mean_ok && var_ok
end

@testset "Move cancellation (target-independent factors cancel)" begin
    @test move_cancellation_test()
end

@testset "Move MH invariance (static-parameter posterior)" begin
    @test move_invariance_test()
end

"""
    move_diversity_skip_test(; N=1_000)

A `Move` with a `diversity_threshold` that is already satisfied (every target
column fully diverse, i.e. all `N` values distinct) must be an exact no-op:
`apply!` should leave the target column bit-for-bit unchanged, without even
needing `state.root`/`state.depth` to be set up for scoring.
"""
function move_diversity_skip_test(; N=1_000)
    Random.seed!(3)
    θ0 = randn(N)
    store = ColumnStore(N)
    state = SMCState(store)
    broadcast_setcol!(store, :θ, identity, (copy(θ0),))

    move = Move([:θ], RW, s -> (0.3,), 0.99)  # threshold below the actual (1.0) diversity
    apply!(move, state)

    return getcol(store, :θ) == θ0
end

"""
    move_diversity_run_test(; T=5, τ0=2.0, σ=1.0, N=50_000, n_moves=100, step_size=0.3, threshold=0.9)

Mirrors `move_invariance_test`, but starts from COLLAPSED particles (all
identical, as if just resampled to a single ancestor) and gates the move with
`diversity_threshold=threshold`. Confirms:
1. The move actually RUNS (particle values change from their initial
   collapsed value) since collapsed particles start at diversity `1/N`, far
   below the threshold.
2. Diversity gating self-limits how far it goes: once the target column's
   diversity crosses `threshold`, further `apply!` calls become no-ops again
   (values stop changing) — this is the expected/intentional trade-off of
   gating on diversity rather than always fully mixing to the target
   distribution, and is worth confirming explicitly rather than assuming.
3. The final diversity is indeed `>= threshold` (the gate's own postcondition).
"""
function move_diversity_run_test(; T=5, τ0=2.0, σ=1.0, N=50_000, n_moves=100, step_size=0.3, threshold=0.9)
    Random.seed!(42)
    y = randn(T) .* σ .+ 1.3

    prior_var = τ0^2
    obs_var = σ^2
    post_var = 1 / (1 / prior_var + T / obs_var)
    post_mean = post_var * (sum(y) / obs_var)

    store = ColumnStore(N)
    state = SMCState(store)
    # Collapsed: every particle starts at the SAME value (diversity = 1/N).
    broadcast_setcol!(store, :θ, identity, (fill(post_mean, N),))
    θ_before = copy(getcol(store, :θ))

    root = Sequence(
        Sample(:θ, NormalKernel, s -> (Ref(0.0), Ref(τ0))),
        (Observe(s -> Ref(y[t]), NormalKernel, s -> (getcol(s.store, :θ), Ref(σ))) for t in 1:T)...,
    )
    state.root = root
    state.depth = T + 1

    move = Move([:θ], RW, s -> (step_size,), threshold)
    for _ in 1:n_moves
        apply!(move, state)
    end

    θ_after_gate_hit = copy(getcol(store, :θ))
    ran = θ_after_gate_hit != θ_before
    gate_satisfied = marginal_diversity(store, [:θ]) >= threshold

    # A few more moves: the gate should now be a no-op (self-limiting).
    for _ in 1:5
        apply!(move, state)
    end
    stopped = getcol(store, :θ) == θ_after_gate_hit

    return ran && gate_satisfied && stopped
end

"""
    move_diversity_marginal_not_joint_test(; N=1_000)

Regression test for the "marginal, not joint, diversity" caveat: builds a
target `α` with very few unique values (collapsed) and a target `β` with all
`N` values distinct. The JOINT tuples `(α[i], β[i])` are then also all
distinct (since `β` alone makes every tuple unique) — a diversity check based
on joint-tuple uniqueness would report full diversity and could wrongly skip
a move that `α` still badly needs. `marginal_diversity` must instead report
the (low) diversity of `α`.
"""
function move_diversity_marginal_not_joint_test(; N=1_000, n_unique_α=5)
    Random.seed!(4)
    α = repeat(collect(1:n_unique_α), inner=N ÷ n_unique_α)
    β = collect(1.0:N)  # all distinct

    store = ColumnStore(N)
    broadcast_setcol!(store, :α, identity, (α,))
    broadcast_setcol!(store, :β, identity, (β,))

    joint_diversity = length(unique(collect(zip(α, β)))) / N
    marg_diversity = marginal_diversity(store, [:α, :β])

    return joint_diversity ≈ 1.0 && marg_diversity ≈ n_unique_α / N
end

@testset "Move diversity gating" begin
    @test move_diversity_skip_test()
    @test move_diversity_run_test()
    @test move_diversity_marginal_not_joint_test()
end

