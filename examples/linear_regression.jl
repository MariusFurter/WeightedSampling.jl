# Linear regression using SMC sampling
using WeightedSampling
using CairoMakie
using Random
Random.seed!(42)

# Generate synthetic data
α = 1.0
β = -0.5
σ = 0.5
xs = 1:10
ys = α .+ β .* xs .+ σ * randn(length(xs))

# Linear regression with importance sampling / resampling.
@smc function linear_regression(xs, ys)
    α ~ Normal(0, 1)
    β ~ Normal(0, 1)

    for (x, y) in zip(xs, ys)
        y => Normal(α + β * x, 1.0)
    end
end

particles, evidence = linear_regression(xs, ys, n_particles=10_000)
describe_particles(particles)


function plot_results(xs, ys, particles, α_true, β_true; title="Linear Regression SMC Sampling")
    fig = Figure(resolution=(400, 300))
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title=title)

    x_range = range(minimum(xs), maximum(xs), length=100)
    y_true = α_true .+ β_true .* x_range

    n_lines = 100
    sampled_particles = sample_particles(particles, n_lines)

    for i in 1:n_lines
        α_sample = sampled_particles.α[i]
        β_sample = sampled_particles.β[i]
        y_sample = α_sample .+ β_sample .* x_range
        lines!(ax, x_range, y_sample, color=(:steelblue, 0.1), linewidth=1)
    end

    # Plot true line
    lines!(ax, x_range, y_true, color=:darkorange, linewidth=3, label="True line")

    # Plot data points
    scatter!(ax, xs, ys, color=:darkred, marker=:xcross, markersize=8, label="Data")

    axislegend(ax, position=:lb)
    fig
end

fig = plot_results(xs, ys, particles, α, β)
display(fig)

# Linear regression with SMC + MCMC moves (requires fewer particles, even for vague priors)
@smc function linear_regression_move(xs, ys)
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)

    for (x, y) in zip(xs, ys)
        y => Normal(α + β * x, 1.0)
        if resampled
            α << autoRW()
            β << autoRW()
        end
    end
end

particles, evidence = linear_regression_move(xs, ys, n_particles=1_000)
describe_particles(particles)

fig = plot_results(xs, ys, particles, α, β)
display(fig)

