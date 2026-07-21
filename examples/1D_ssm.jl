using WeightedSampling
using Random
using CairoMakie

Random.seed!(7)

@model function ssm(obs)
    x{1} .= 0.0
    v .= 0.0
    for (t, o) in enumerate(obs)
        x{t + 1} .= x{t} + v
        dv ~ Normal(0.0, 0.1)
        v .= v + dv
        o => Normal(x{t + 1}, 1.0)
    end
end

T = 50
x_0, v_0 = 0.0, 0.0
xs_true, vs_true, obs = [x_0], [v_0], Float64[]
for t in 1:T
    push!(obs, xs_true[end] + 1.0 * randn())
    new_x = xs_true[t] + vs_true[t]
    new_v = vs_true[t] + 0.1 * randn()
    push!(xs_true, new_x)
    push!(vs_true, new_v)
end

n_particles = 1000
model = ssm(obs)
state = SMCState(n_particles)
run!(model, state)

# variable access
state[:x_1]
# summary statistics
describe(state) 
# weighted expectations
@E((x_1,x_2) -> x_1 + x_2, state)
# draw unweighted samples
sample(state, 100)

trajectories = sample(state, 100)

# =============================================================================
# Plot: sampled trajectories with true path and observations overlaid
# =============================================================================

# Dynamic-variable column names x_1, ..., x_{T+1} (see `dynname`), each
# holding a scalar position for every particle.
xnames = [Symbol(:x_, i) for i in 1:(T + 1)]

times = 0:T
obs_times = 1:T

fig = Figure(; resolution=(350, 250))
ax = Axis(fig[1, 1];
    title="Posterior trajectories",
    xlabel="t",
    ylabel="x",
)

# Semi-transparent posterior draws ("spaghetti plot") show the range of
# plausible trajectories implied by the posterior.
for row in eachrow(trajectories)
    pts = [row[c] for c in xnames]
    lines!(ax, times, pts;
        color=(:steelblue, 0.07), linewidth=2)
end

# True simulated trajectory, highlighted in front of the spaghetti plot.
lines!(ax, times, xs_true;
    color=:crimson, linewidth=3, label="True path")

# Observed data points.
scatter!(ax, obs_times, obs;
    color=:black, markersize=8, label="Observations")

fig
save(joinpath(@__DIR__, "plots", "1D_ssm.png"), fig; px_per_unit=3.0)

