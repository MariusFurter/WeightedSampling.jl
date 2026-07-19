using WeightedSampling
using Random
using CairoMakie

# =============================================================================
# Bayesian linear regression via Sequential Monte Carlo
#
# We fit y = α + β*x + noise using WeightedSampling.jl's `@model` macro,
# then visualize the posterior over regression lines together with the
# observed data.
# =============================================================================

Random.seed!(42)

# --- Model definition ---

@model function linear_regression(xs, ys)
    α ~ Normal(0.0, 10.0)
    β ~ Normal(0.0, 10.0)
    for (x, y) in zip(xs, ys)
        y => Normal(α + β * x, 1.0)
        if resampled
            α << autoRW()
            β << autoRW()
        end
    end
end

# --- Simulate synthetic data ---

α_true, β_true, σ = 1.0, -0.5, 0.5
xs = 1:10
ys = α_true .+ β_true .* xs .+ σ * randn(length(xs))

# --- Run SMC inference ---

n_particles = 1000
model = linear_regression(xs, ys)
state = SMCState(n_particles)
run!(model, state)

# Summary statistics (mean, std, histogram) for each particle variable.
describe(state)

# Equally-weighted posterior sample, e.g. for inspecting the α marginal.
result = sample(state, 1000)
result[!, :α]

# =============================================================================
# Plot: posterior regression lines with observed data overlaid
# =============================================================================

# Draw a handful of (α, β) pairs from the posterior to visualize uncertainty
# as a spread of candidate regression lines.
n_lines = 200
posterior = sample(state, n_lines)

# Posterior mean line, computed as a weighted expectation over ALL particles
α_mean = @E(α -> α, state)
β_mean = @E(β -> β, state)

x_grid = range(minimum(xs), maximum(xs); length=100)

fig = Figure(; resolution=(500, 400))
ax = Axis(fig[1, 1];
    title="Posterior Regression Lines",
    xlabel="x",
    ylabel="y",
)

# Semi-transparent posterior draws ("spaghetti plot") show the range of
# plausible regression lines implied by the posterior.
for row in eachrow(posterior)
    lines!(ax, x_grid, row.α .+ row.β .* x_grid;
        color=(:steelblue, 0.05), linewidth=2)
end

# Posterior mean line, highlighted in front of the spaghetti plot.
lines!(ax, x_grid, α_mean .+ β_mean .* x_grid;
    color=:crimson, linewidth=3, label="Posterior mean")

# Observed data points.
scatter!(ax, collect(xs), ys;
    color=:black, markersize=12, label="Observed data")

axislegend(ax; position=:rt)

fig
save(joinpath(@__DIR__, "plots", "linear_regression.png"), fig; px_per_unit=2.0)