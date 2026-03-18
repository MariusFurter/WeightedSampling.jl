# Eight Schools Benchmark — WeightedSampling.jl (SMC)
#
# Hierarchical model from Gelman et al. (2003) "Bayesian Data Analysis", Sec 5.5.
#
# Usage: julia --project benchmarks/eight_schools.jl

using WeightedSampling
using Distributions
using Random
using Printf
using Statistics

# Data
J = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

@model function eight_schools(J, y, σ)
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

function benchmark_eight_schools()
    n_particles_list = [1_000, 5_000, 10_000]
    n_runs = 5

    println("="^60)
    println("Eight Schools Benchmark — WeightedSampling.jl (SMC)")
    println("="^60)
    @printf("Runs per config: %d (median reported)\n", n_runs)

    # Warmup (compilation)
    Random.seed!(42)
    eight_schools(J, y, σ; n_particles=100, show_progress=false)

    for n_particles in n_particles_list
        println("\n--- n_particles = $n_particles ---")

        times = Float64[]
        local particles, evidence
        for run in 1:n_runs
            Random.seed!(42 + run)
            t = @elapsed begin
                particles, evidence = eight_schools(J, y, σ;
                    n_particles=n_particles, show_progress=false)
            end
            push!(times, t)
        end

        med_time = median(times)

        # Posterior summaries
        μ_mean = @E(μ -> μ, particles)
        μ_std = sqrt(@E(μ -> μ^2, particles) - μ_mean^2)

        τ_mean = @E(τ -> τ, particles)
        τ_std = sqrt(max(@E(τ -> τ^2, particles) - τ_mean^2, 0.0))

        θ_means = Float64[]
        for j in 1:J
            θ_j = [θ[j] for θ in particles.θ]
            w = exp_norm(particles.weights)
            push!(θ_means, sum(θ_j .* w))
        end

        n_unique_μ = length(unique(round.(particles.μ, digits=6)))

        @printf("Median time:  %.4f s  (range: %.4f – %.4f)\n",
            med_time, minimum(times), maximum(times))
        @printf("Log evidence: %.4f\n", evidence)
        @printf("μ:            %.2f ± %.2f\n", μ_mean, μ_std)
        @printf("τ:            %.2f ± %.2f\n", τ_mean, τ_std)
        @printf("θ means:      %s\n", join([@sprintf("%.2f", m) for m in θ_means], ", "))
        @printf("Unique μ:     %d / %d\n", n_unique_μ, n_particles)
    end

    println("\n" * "="^60)
end

benchmark_eight_schools()
