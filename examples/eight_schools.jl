using WeightedSampling
using Random
using CairoMakie

Random.seed!(42)

@model function eight_schools(J, y, σ)
    μ ~ Normal(0.0, 5.0)
    τ ~ Exponential(5.0)
    θ .= zeros(J)
    for j in 1:J
        θ[j] ~ Normal(μ, τ)
        y[j] => Normal(θ[j], σ[j])
        μ << autoRW(; diversity=0.9)
        τ << autoRW(1e-3, (0.0, Inf); diversity=0.9)
    end
end

J = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

n_particles = 1000
model = eight_schools(J, y, σ)
state = SMCState(n_particles)
run!(model, state)

describe(state)

# =============================================================================
# Plot: caterpillar (forest) plot of the posterior over μ and each θ[j]
# =============================================================================
#
# θ is stored as one array-valued column (`state[:θ]`), an N-vector of
# J-vectors (one per particle). We extract weighted posterior mean/std per
# school by pulling out the j-th component across all particles.

weights = exp_norm(state.weights)

μ_mean = @E(μ -> μ, state)
μ_std = sqrt(@E(μ -> μ^2, state) - μ_mean^2)

θ_samples = state[:θ]
θ_mean = [expectation([p[j] for p in θ_samples], state.weights) for j in 1:J]
θ_std = [sqrt(expectation([p[j]^2 for p in θ_samples], state.weights) - θ_mean[j]^2) for j in 1:J]

# Row labels, top-to-bottom: μ (pooled mean), then each school's θ[j].
labels = vcat("μ", ["School $j" for j in 1:J])
post_mean = vcat(μ_mean, θ_mean)
post_std = vcat(μ_std, θ_std)
row = length(labels):-1:1

fig = Figure(; resolution=(600, 450))
ax = Axis(fig[1, 1];
    title="Eight Schools: Posterior Estimates",
    xlabel="Effect size",
    yticks=(row, labels),
)

# Raw per-school observations (y ± σ), offset slightly below each posterior
# estimate, for comparison against the (partially pooled) posterior.
rangebars!(ax, row[2:end] .- 0.15, y .- σ, y .+ σ;
    direction=:x, color=:gray, whiskerwidth=8)
scatter!(ax, y, row[2:end] .- 0.15;
    color=:gray, markersize=10, label="Observed (y ± σ)")

# Posterior mean ± 1 std for μ and each θ[j].
rangebars!(ax, row, post_mean .- post_std, post_mean .+ post_std;
    direction=:x, color=:steelblue, whiskerwidth=10)
scatter!(ax, post_mean, row;
    color=:crimson, markersize=12, label="Posterior mean ± 1σ")

vlines!(ax, [μ_mean]; color=(:crimson, 0.3), linestyle=:dash)

axislegend(ax; position=:rb)

fig
save(joinpath(@__DIR__, "plots", "eight_schools.png"), fig; px_per_unit=2.0)


