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
