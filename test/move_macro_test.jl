using Test
using Random
using Distributions
using DataFrames


"""
    linear_regression_mh_macro_test(; n_particles=10_000, atol_slope=0.3, atol_intercept=0.3)

Macro-built equivalent of `test/linear_regression_mh_test.jl`, exercising
`@model`'s `<<` support end-to-end: a static-parameter move gated by
`if resampled`, the idiomatic `Cond`-gated-`Move` pattern.

    α ~ Normal(0, 5)
    β ~ Normal(0, 5)
    for (x, y) in data
        y => Normal(α + β * x, 0.5)
        if resampled
            (α, β) << RW(0.1)
        end
    end

Must be run with [`run!`](@ref) (not plain `apply!`), so `state.root` is set
for the `Move`'s `score!` fold.
"""
function linear_regression_mh_macro_test(; n_particles=10_000, atol_slope=0.3, atol_intercept=0.3)
    Random.seed!(42)

    true_α = -1.0
    true_β = 2.0
    noise_std = 0.5
    n_points = 10

    xs = collect(range(0, 10, length=n_points))
    ys = true_α .+ true_β .* xs .+ noise_std .* randn(n_points)
    data = collect(zip(xs, ys))

    @model function linear_regression(data)
        α ~ Normal(0.0, 5.0)
        β ~ Normal(0.0, 5.0)
        for (x, y) in data
            y => Normal(α + β * x, 0.5)
            if resampled
                (α, β) << RW(0.1)
            end
        end
    end

    model = linear_regression(data; kernels=(Normal=NormalKernel,), proposals=(RW=RW,))
    state = SMCState(ColumnStore(n_particles))
    run!(model, state)

    w = exp_norm(state.weights)
    est_α = sum(getcol(state.store, :α) .* w)
    est_β = sum(getcol(state.store, :β) .* w)

    α_ok = isapprox(est_α, true_α, atol=atol_intercept)
    β_ok = isapprox(est_β, true_β, atol=atol_slope)

    return α_ok && β_ok
end

@testset "Macro: linear regression with << MH moves" begin
    @test linear_regression_mh_macro_test()
end
