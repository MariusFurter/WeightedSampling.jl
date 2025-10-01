# Multi-level regression using SMC sampling
using WeightedSampling
using CairoMakie
using StatsBase
using Random
Random.seed!(42)

# 8 Schools example from Gelman et al. (2003) "Bayesian Data Analysis", 2nd edition, Chapter 5.5

J = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]  # estimated treatment effects
σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]  # standard errors of effect estimates

@smc function eight_schools(J, y, σ)
    μ ~ Normal(0, 5)
    τ ~ Exponential(5)

    θ .= zeros(J)
    for j in 1:J
        θ[j] ~ Normal(μ, τ)
        y[j] => Normal(θ[j], σ[j])
        if resampled
            μ << autoRW()
            τ << autoRW()
        end
    end
end

samples, evidence = eight_schools(J, y, σ, n_particles=1_000)
describe_particles(samples)

function plot_theta_densities(samples, lower, upper)
    fig = Figure(resolution=(400, 400))
    ax = Axis(fig[1, 1], xlabel="θ", ylabel="Density", title="Posterior Densities of θ")

    for j in 1:J
        theta_j = [θ[j] for θ in samples.θ]
        density!(ax, theta_j, weights=exp_norm(samples.weights), label="θ[$j]",
            offset=(j - 1) * 0.2, direction=:x)
    end

    xlims!(ax, lower, upper)

    axislegend(ax)
    return fig
end

fig = plot_theta_densities(samples, -20, 30)
display(fig)

