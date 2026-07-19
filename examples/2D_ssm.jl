using WeightedSampling
using Random
using CairoMakie

Random.seed!(42)

@model function ssm(obs)
    I2 = [1.0 0.0; 0.0 1.0]
    x{1} .= [0.0, 0.0]
    v .= [1.0, 0.0]
    for (i, o) in enumerate(obs)
        x{i + 1} .= x{i} + v
        dv ~ MvNormal([0.0, 0.0], 0.1 * I2)
        v .= v + dv
        o => MvNormal(x{i + 1}, 0.5 * I2)
    end
end

T = 50
x_0, v_0 = [0.0, 0.0], [1.0, 0.0]
xs_true, vs_true, obs = [x_0], [v_0], Vector{Float64}[]
for t in 1:T
    push!(obs, xs_true[end] + 0.5 * randn(2))
    new_x = xs_true[t] + vs_true[t]
    new_v = vs_true[t] + 0.1 * randn(2)
    push!(xs_true, new_x)
    push!(vs_true, new_v)
end

n_particles = 1000
model = ssm(obs)
state = SMCState(n_particles)
run!(model, state)

trajectories = sample(state, 100)

# =============================================================================
# Plot: sampled trajectories with true path and observations overlaid
# =============================================================================

# Dynamic-variable column names x_1, ..., x_{T+1} (see `dynname`), each
# holding a 2D position for every particle.
xnames = [Symbol(:x_, i) for i in 1:(T + 1)]

fig = Figure(; resolution=(500, 400))
ax = Axis(fig[1, 1];
    title="Posterior Trajectories",
    xlabel="x₁",
    ylabel="x₂",
)

# Semi-transparent posterior draws ("spaghetti plot") show the range of
# plausible trajectories implied by the posterior.
for row in eachrow(trajectories)
    pts = [row[c] for c in xnames]
    lines!(ax, getindex.(pts, 1), getindex.(pts, 2);
        color=(:steelblue, 0.05), linewidth=2)
end

# True simulated trajectory, highlighted in front of the spaghetti plot.
lines!(ax, getindex.(xs_true, 1), getindex.(xs_true, 2);
    color=:crimson, linewidth=3, label="True path")

# Observed data points.
scatter!(ax, getindex.(obs, 1), getindex.(obs, 2);
    color=:black, markersize=8, label="Observations")

axislegend(ax; position=:rt)

fig
save(joinpath(@__DIR__, "plots", "2D_ssm.png"), fig; px_per_unit=2.0)

