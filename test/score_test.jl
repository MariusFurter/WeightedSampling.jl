using Test
using Random
using Distributions


"""
    score_logpdf_unit_test(; N=1_000)

Hand-builds a tiny tree (no `Move`):

    θ ~ Normal(0, 1)          # depth 0 -> 1
    x .= θ                    # depth 1 -> 2 (deterministic, no score contribution)
    y => Normal(x, 0.5)       # depth 2 -> 3 (observation)

and checks that `score_logpdf(state, [:θ], target_depth)` equals the
hand-computed sum of the relevant `logpdf`s for every `target_depth` in
`0:3`, confirming both the accumulated value and the early-termination depth
cutoff (raising `target_depth` includes exactly the expected extra factor).
"""
function score_logpdf_unit_test(; N=1_000)
    Random.seed!(42)

    root = Sequence(
        Sample(:θ, NormalKernel, state -> (Ref(0.0), Ref(1.0))),
        Assign(:x, state -> getcol(state.store, :θ)),
        Observe(state -> Ref(1.5), NormalKernel, state -> (getcol(state.store, :x), Ref(0.5))),
    )

    state = SMCState(ColumnStore(N))
    run!(root, state)

    θ = getcol(state.store, :θ)
    x = getcol(state.store, :x)

    # target_depth = 0: nothing scored yet.
    s0 = score_logpdf(state, [:θ], 0)
    ok0 = all(iszero, s0)

    # target_depth = 1: only the θ ~ Normal(0,1) prior draw.
    s1 = score_logpdf(state, [:θ], 1)
    expected1 = logpdf.(Normal(0, 1), θ)
    ok1 = isapprox(s1, expected1)

    # target_depth = 2: Assign contributes nothing extra (still just the prior).
    s2 = score_logpdf(state, [:θ], 2)
    ok2 = isapprox(s2, expected1)

    # target_depth = 3: prior + observation.
    s3 = score_logpdf(state, [:θ], 3)
    expected3 = expected1 .+ logpdf.(Normal.(x, 0.5), 1.5)
    ok3 = isapprox(s3, expected3)

    return ok0 && ok1 && ok2 && ok3
end

@testset "score! unit test (depth cutoff + accumulation)" begin
    @test score_logpdf_unit_test()
end
