using Test
using Random
using Distributions
using DataFrames


"""
    linear_regression_dyn_mh_test(; n_particles=10_000, atol_slope=0.3, atol_intercept=0.3)

Same model/tolerances as `move_macro_test.jl`'s `linear_regression_mh_macro_test`,
but `β` is a dynamic-variable family member `β{1}` instead of a plain particle
variable, and the `Move` target is the MIXED tuple `(α, β{1})` — exercises
`<<` accepting a dynamic-variable family target alongside a plain one.
"""
function linear_regression_dyn_mh_test(; n_particles=10_000, atol_slope=0.3, atol_intercept=0.3)
    Random.seed!(42)

    true_α = -1.0
    true_β = 2.0
    noise_std = 0.5
    n_points = 10

    xs = collect(range(0, 10, length=n_points))
    ys = true_α .+ true_β .* xs .+ noise_std .* randn(n_points)
    data = collect(zip(xs, ys))

    @model function linear_regression_dyn(data)
        α ~ Normal(0.0, 5.0)
        β{1} ~ Normal(0.0, 5.0)
        for (x, y) in data
            y => Normal(α + β{1} * x, 0.5)
            if resampled
                (α, β{1}) << RW(0.1)
            end
        end
    end

    model = linear_regression_dyn(data; kernels=(Normal=NormalKernel,), proposals=(RW=RW,))
    state = SMCState(ColumnStore(n_particles))
    run!(model, state)

    w = exp_norm(state.weights)
    est_α = sum(getcol(state.store, :α) .* w)
    est_β = sum(getcol(state.store, :β_1) .* w)

    α_ok = isapprox(est_α, true_α, atol=atol_intercept)
    β_ok = isapprox(est_β, true_β, atol=atol_slope)

    return α_ok && β_ok
end

@testset "Macro: << Move target compatible with x{i}" begin
    @test linear_regression_dyn_mh_test()
end

@testset "Macro: << Move rejects value-level accessor targets" begin
    # `x[e]`/`x.p` are element-level mutations, not valid whole-column Move
    # targets — must be a clear macro-expansion-time error.
    @test_throws Exception eval(:(@model function bad_move_ref()
        v .= [1.0, 2.0]
        v[1] << RW(0.1)
    end))

    @test_throws Exception eval(:(@model function bad_move_prop()
        p .= 1.0
        p.x << RW(0.1)
    end))

    # Mixed tuple containing an accessor target should also error.
    @test_throws Exception eval(:(@model function bad_move_tuple()
        a ~ Normal(0, 1)
        v .= [1.0, 2.0]
        (a, v[1]) << RW(0.1)
    end))
end
