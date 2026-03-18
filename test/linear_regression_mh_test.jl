### Linear regression with MH moves test
# Bayesian linear regression: y = α + β*x + noise
# Uses MH random walk moves after resampling to maintain particle diversity.
#
# Ported from WeightedSampling.torch examples/linear_regression.py

function linear_regression_mh_test(; n_particles=10_000, atol_slope=0.3, atol_intercept=0.3)
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

    particles, evidence = linear_regression(data; n_particles=n_particles, show_progress=false)

    est_α = @E(α -> α, particles)
    est_β = @E(β -> β, particles)

    α_ok = isapprox(est_α, true_α, atol=atol_intercept)
    β_ok = isapprox(est_β, true_β, atol=atol_slope)

    return α_ok && β_ok
end
