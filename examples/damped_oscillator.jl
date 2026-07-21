using WeightedSampling
using Random
using CairoMakie
using Distributions
using Statistics: std

Random.seed!(42)

# --- Model definition ---

oscillator(t, A, ω, γ, ϕ) = A * exp(-γ * t) * cos(ω * t + ϕ)

A_true = 3.0
γ_true = 0.3
ω_true = 2.5
ϕ_true = 0.5
σ_true = 1.0

# Observations
t_obs = range(0, 8, length=60)
y_true = oscillator.(t_obs, A_true, ω_true, γ_true, ϕ_true)
y_obs = y_true + σ_true * randn(length(t_obs))

HalfNormal = WeightedKernel(
    (σ) -> rand(Truncated(Normal(0.0, σ), 0.0, Inf)),
    nothing,
    (σ, x) -> logpdf(Truncated(Normal(0.0, σ), 0.0, Inf), x)
)

@model function damped_oscillator(t_obs, y_obs)
    A ~ HalfNormal(5)
    ω ~ HalfNormal(5)
    γ ~ HalfNormal(1)
    ϕ ~ Uniform(-π, π)
    σ ~ HalfNormal(1)

    for (t, y) in zip(t_obs, y_obs)
        y => Normal(oscillator(t, A, ω, γ, ϕ), σ)
        (A, ω, γ, σ) << autoRW(1e-3, (0.0, Inf), diversity=0.9)
        ϕ << autoRW(1e-3, (-π, π), diversity=0.9)
    end

end

# --- Run particle filter ---

n_particles = 1000
model = damped_oscillator(t_obs, y_obs, kernels=(HalfNormal=HalfNormal,))
state = SMCState(n_particles)
run!(model, state)

describe(state)

post_samples = sample(state, 100)

# --- Plot posterior trajectories with observed points and true curve overlaid ---

t_plot = range(first(t_obs), last(t_obs), length=300)

fig = Figure(resolution=(350, 250))
ax = Axis(fig[1, 1], xlabel="t", ylabel="y", title="Posterior trajectories")

for row in eachrow(post_samples)
    lines!(ax, t_plot, oscillator.(t_plot, row.A, row.ω, row.γ, row.ϕ),
        color=(:steelblue, 0.15))
end

lines!(ax, t_plot, oscillator.(t_plot, A_true, ω_true, γ_true, ϕ_true),
    color=:crimson, linewidth=2, label="true curve")
scatter!(ax, t_obs, y_obs, color=:black, markersize=8, label="observations")

save(joinpath(@__DIR__, "plots", "damped_oscillator.png"), fig)
fig

# --- Ridge plot of marginal posterior densities, with true values marked ---

density_samples = sample(state, 5000)

params = [
    (:A, A_true),
    (:ω, ω_true),
    (:γ, γ_true),
    (:ϕ, ϕ_true),
    (:σ, σ_true),
]

# Simple Gaussian KDE on a grid, normalized to unit peak height.
function kde_grid(x; npoints=200)
    lo, hi = extrema(x)
    pad = 0.1 * (hi - lo)
    grid = range(lo - pad, hi + pad, length=npoints)
    bw = 1.06 * std(x) * length(x)^(-1 / 5)
    dens = [sum(pdf(Normal(g, bw), xi) for xi in x) / length(x) for g in grid]
    return grid, dens ./ maximum(dens)
end

# Linear interpolation of `dens` (evaluated on `grid`) at `x`.
function interp(grid, dens, x)
    x <= first(grid) && return first(dens)
    x >= last(grid) && return last(dens)
    j = searchsortedlast(grid, x)
    t = (x - grid[j]) / (grid[j+1] - grid[j])
    return dens[j] + t * (dens[j+1] - dens[j])
end

# Stack densities on a single unified y axis (rows offset vertically).
fig_ridge = Figure(resolution=(350, 200))
ax = Axis(fig_ridge[1, 1])

row_height = 0.9
spacing = 1.0
offsets = [(length(params) - i) * spacing for i in 1:length(params)]

for ((name, true_val), offset) in zip(params, offsets)
    grid, dens = kde_grid(density_samples[!, name])
    dens = dens .* row_height
    band!(ax, grid, fill(offset, length(grid)), offset .+ dens,
        color=(:steelblue, 0.4))
    lines!(ax, grid, offset .+ dens, color=:steelblue, linewidth=1.5)

    true_height = interp(grid, dens, true_val)
    lines!(ax, [true_val, true_val], [offset, offset + true_height],
        color=:crimson, linewidth=2, linestyle=:dash)
end

ax.yticks = (offsets, string.(first.(params)))

save(joinpath(@__DIR__, "plots", "damped_oscillator_ridge.png"), fig_ridge, px_per_unit=3.0)
fig_ridge