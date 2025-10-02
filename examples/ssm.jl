# 2D State space model
using WeightedSampling
using DataFrames
using CairoMakie
using Random
Random.seed!(42)

# Generate trajectory
function update(x, v)
    new_x = x + v
    new_v = v + 0.1 * randn(2)
    return new_x, new_v
end

T = 50

x_0 = [0.0, 0.0]
v_0 = [1.0, 0.0]
xs = [x_0]
vs = [v_0]
obs = []

for t in 1:T
    push!(obs, xs[end] + 0.5 * randn(2))
    new_x, new_v = update(xs[t], vs[t])
    push!(xs, new_x)
    push!(vs, new_v)
end

# State space model with SMC
@smc function ssm(obs)
    I = [1 0
        0 1]

    x1 .= [0.0, 0.0]
    v .= [1.0, 0.0]
    for (i, o) in enumerate(obs)
        x{i + 1} .= x{i} + v
        dv ~ MvNormal([0, 0], 0.1 * I)
        v .= v + dv
        o => MvNormal(x{i + 1}, 0.5 * I)
    end
end

particles, evidence = ssm(obs, n_particles=1_000)
describe_particles(particles)

# Plot sampled trajectories
function plot_trajectory(xs, obs, filtered_particles=nothing)
    fig = Figure(resolution=(500, 400))
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="2D State Space Model")

    # Plot true trajectory
    true_x1 = [x[1] for x in xs]
    true_x2 = [x[2] for x in xs]
    lines!(ax, true_x1, true_x2, color=:steelblue, linewidth=2, label="True trajectory")

    # Plot observations
    obs_x1 = [o[1] for o in obs]
    obs_x2 = [o[2] for o in obs]
    scatter!(ax, obs_x1, obs_x2, marker=:cross, color=:darkorange, markersize=10, label="Observations")

    # Plot filtered states if provided
    if filtered_particles !== nothing
        x_cols = [name for name in names(filtered_particles) if startswith(string(name), "x")]
        x_cols = sort(x_cols, by=x -> parse(Int, string(x)[2:end]))

        if !isempty(x_cols)
            # Sample a subset of particles to plot
            n_to_plot = min(200, nrow(filtered_particles))
            sampled_particles = sample_particles(filtered_particles, n_to_plot)

            for i in 1:n_to_plot
                # Extract trajectory for particle i
                traj_x1 = [sampled_particles[i, col][1] for col in x_cols]
                traj_x2 = [sampled_particles[i, col][2] for col in x_cols]
                lines!(ax, traj_x1, traj_x2, color=(:green, 0.05), linewidth=1)
            end

            # Add a single label for all particle trajectories
            lines!(ax, Float64[], Float64[], color=(:green, 0.3), linewidth=1, label="Sampled trajectories at t=$T")
        end
    end

    axislegend(ax, position=:ct)
    return fig
end

fig = plot_trajectory(xs, obs, particles)
display(fig)
