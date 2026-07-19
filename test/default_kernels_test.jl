using Test
using Random
using Statistics

"""
Confirms `@model` resolves `~`/`<<` against `WeightedSampling.default_kernels`/
`default_proposals` with NO explicit `kernels=`/`proposals=` kwarg passed at
all. Every other test file in this suite passes an explicit `kernels=
(Normal=NormalKernel,)` (see `models.jl`), so without this file the automatic
default-merge path (`merge(WeightedSampling.default_kernels, kernels)` inside
every generated `@model` function, see `rewrites.jl`) is never actually
exercised end-to-end.
"""
@model function random_walk_default(T::Int)
    x ~ Normal(0, 1)
    for t in 1:T
        x ~ Normal(x, 1)
    end
end

@model function linear_regression_default(data)
    α ~ Normal(0.0, 5.0)
    β ~ Normal(0.0, 5.0)
    for (x, y) in data
        y => Normal(α + β * x, 0.5)
        if resampled
            (α, β) << RW(0.1)
        end
    end
end

@testset "Default kernels: ~ resolves Normal with no explicit kernels kwarg" begin
    Random.seed!(42)
    T = 10
    N = 100_000

    model = random_walk_default(T)
    state = SMCState(ColumnStore(N))
    apply!(model, state)

    xs = getcol(state.store, :x)
    @test isapprox(mean(xs), 0.0, atol=0.15)
    @test isapprox(var(xs), T + 1, rtol=0.05)
end

@testset "Default proposals: << resolves RW with no explicit proposals kwarg" begin
    Random.seed!(42)

    true_α, true_β, noise_std, n_points = -1.0, 2.0, 0.5, 10
    xs = collect(range(0, 10, length=n_points))
    ys = true_α .+ true_β .* xs .+ noise_std .* randn(n_points)
    data = collect(zip(xs, ys))

    model = linear_regression_default(data)
    state = SMCState(ColumnStore(10_000))
    run!(model, state)

    w = exp_norm(state.weights)
    est_α = sum(getcol(state.store, :α) .* w)
    est_β = sum(getcol(state.store, :β) .* w)

    @test isapprox(est_α, true_α, atol=0.3)
    @test isapprox(est_β, true_β, atol=0.3)
end
