# Hierarchical Regression Benchmark — WeightedSampling.jl (SMC)
#
# Model:
#   μ_α ~ N(0, 10), μ_β ~ N(0, 10)        (population means)
#   σ_α ~ LogNormal(0, 1), σ_β ~ LogNormal(0, 1)  (population scales, via log-reparam)
#   α_j ~ N(μ_α, σ_α)  for j = 1,...,J     (group intercepts)
#   β_j ~ N(μ_β, σ_β)  for j = 1,...,J     (group slopes)
#   y_ij ~ N(α_j + β_j * x_ij, 1.0)       (observations, known noise)
#
# J = 50 groups, N = 50 observations per group, 2500 total.
#
# Usage: julia --project benchmarks/hierarchical_regression.jl

using WeightedSampling
using Distributions
using Random
using Printf
using Statistics

const TRUE_MU_ALPHA = 2.0
const TRUE_MU_BETA = -1.0
const TRUE_SIGMA_ALPHA = 1.5
const TRUE_SIGMA_BETA = 0.8
const SIGMA_Y = 1.0

function generate_data(J, N)
    Random.seed!(42)

    alphas = TRUE_MU_ALPHA .+ TRUE_SIGMA_ALPHA .* randn(J)
    betas = TRUE_MU_BETA .+ TRUE_SIGMA_BETA .* randn(J)

    # Store as vector of vectors of (x, y) tuples per group
    group_data = Vector{Vector{Tuple{Float64,Float64}}}()
    for j in 1:J
        xs = randn(N)
        ys = alphas[j] .+ betas[j] .* xs .+ SIGMA_Y .* randn(N)
        push!(group_data, collect(zip(xs, ys)))
    end

    return group_data, alphas, betas
end

@smc function hier_regression(J, group_data)
    μ_α ~ Normal(0, 10)
    μ_β ~ Normal(0, 10)
    log_σ_α ~ Normal(0, 1)
    log_σ_β ~ Normal(0, 1)

    α .= zeros(J)
    β .= zeros(J)

    for j in 1:J
        α[j] ~ Normal(μ_α, exp(log_σ_α))
        β[j] ~ Normal(μ_β, exp(log_σ_β))

        for (x_i, y_i) in group_data[j]
            y_i => Normal(α[j] + β[j] * x_i, 1.0)
            if resampled
                μ_α << autoRW()
                μ_β << autoRW()
                log_σ_α << autoRW()
                log_σ_β << autoRW()
            end
        end
    end
end

function benchmark_hier_regression()
    J = 50
    N = 50
    n_particles_list = [1_000, 5_000, 10_000]
    n_runs = 5

    group_data, true_alphas, true_betas = generate_data(J, N)

    println("="^60)
    println("Hierarchical Regression — WeightedSampling.jl (SMC)")
    println("="^60)
    @printf("Groups: %d, Obs/group: %d, Total obs: %d\n", J, N, J * N)
    @printf("True μ_α=%.1f, μ_β=%.1f, σ_α=%.1f, σ_β=%.1f\n",
        TRUE_MU_ALPHA, TRUE_MU_BETA, TRUE_SIGMA_ALPHA, TRUE_SIGMA_BETA)
    @printf("Runs per config: %d (median reported)\n", n_runs)

    # Warmup (compilation)
    Random.seed!(0)
    hier_regression(J, group_data; n_particles=100, show_progress=false)

    for n_particles in n_particles_list
        println("\n--- n_particles = $n_particles ---")

        times = Float64[]
        local particles, evidence
        for run in 1:n_runs
            Random.seed!(42 + run)
            t = @elapsed begin
                particles, evidence = hier_regression(J, group_data;
                    n_particles=n_particles, show_progress=false)
            end
            push!(times, t)
        end

        med_time = median(times)

        # Population-level posteriors
        μ_α_est = @E(μ_α -> μ_α, particles)
        μ_β_est = @E(μ_β -> μ_β, particles)
        σ_α_est = @E(log_σ_α -> exp(log_σ_α), particles)
        σ_β_est = @E(log_σ_β -> exp(log_σ_β), particles)

        # Group-level recovery
        w = exp_norm(particles.weights)
        α_means = [sum([a[j] for a in particles.α] .* w) for j in 1:J]
        β_means = [sum([b[j] for b in particles.β] .* w) for j in 1:J]
        α_rmse = sqrt(mean((α_means .- true_alphas) .^ 2))
        β_rmse = sqrt(mean((β_means .- true_betas) .^ 2))

        @printf("Median time:  %.4f s  (range: %.4f – %.4f)\n",
            med_time, minimum(times), maximum(times))
        @printf("Log evidence: %.4f\n", evidence)
        @printf("μ_α:          %.3f  (true: %.1f)\n", μ_α_est, TRUE_MU_ALPHA)
        @printf("μ_β:          %.3f  (true: %.1f)\n", μ_β_est, TRUE_MU_BETA)
        @printf("σ_α:          %.3f  (true: %.1f)\n", σ_α_est, TRUE_SIGMA_ALPHA)
        @printf("σ_β:          %.3f  (true: %.1f)\n", σ_β_est, TRUE_SIGMA_BETA)
        @printf("α RMSE:       %.4f\n", α_rmse)
        @printf("β RMSE:       %.4f\n", β_rmse)
    end

    println("\n" * "="^60)
end

benchmark_hier_regression()
