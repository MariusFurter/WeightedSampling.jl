# simulate.jl
#
# Simulates ground truth data for a hierarchical (partial-pooling) linear
# regression model:
#
#     mu_alpha ~ Normal(0, 10)
#     tau_alpha ~ Exponential(1)
#     beta ~ Normal(0, 10)
#     sigma ~ Exponential(1)
#     alpha[j] ~ Normal(mu_alpha, tau_alpha)      for j in 1:J
#     y[i] ~ Normal(alpha[group[i]] + beta * x[i], sigma)
#
# Writes `data.csv` (columns: group,x,y; 1-indexed group) and
# `true_params.txt` (key=value lines, including the true global parameters and
# the per-group true `alpha_j`) into `data_dir`.
#
# Usage: `julia simulate.jl <J> <n_obs_per_group> <seed> <data_dir>`
using Random
using Printf

function simulate_hierarchical_regression(J::Int, n_obs::Int; seed::Int=42,
        mu_alpha::Float64=5.0, tau_alpha::Float64=2.0, beta::Float64=3.0, sigma::Float64=1.0)
    Random.seed!(seed)
    alpha = mu_alpha .+ tau_alpha .* randn(J)

    groups = Int[]
    xs = Float64[]
    ys = Float64[]
    for j in 1:J
        for _ in 1:n_obs
            x = randn()
            y = alpha[j] + beta * x + sigma * randn()
            push!(groups, j)
            push!(xs, x)
            push!(ys, y)
        end
    end
    return groups, xs, ys, alpha, (mu_alpha=mu_alpha, tau_alpha=tau_alpha, beta=beta, sigma=sigma)
end

function write_dataset(data_dir::AbstractString, J::Int, n_obs::Int, seed::Int)
    mkpath(data_dir)
    groups, xs, ys, alpha, truth = simulate_hierarchical_regression(J, n_obs; seed=seed)

    open(joinpath(data_dir, "data.csv"), "w") do io
        println(io, "group,x,y")
        for i in eachindex(groups)
            @printf(io, "%d,%.10f,%.10f\n", groups[i], xs[i], ys[i])
        end
    end

    open(joinpath(data_dir, "true_params.txt"), "w") do io
        @printf(io, "J=%d\n", J)
        @printf(io, "n_obs=%d\n", n_obs)
        @printf(io, "mu_alpha=%.10f\n", truth.mu_alpha)
        @printf(io, "tau_alpha=%.10f\n", truth.tau_alpha)
        @printf(io, "beta=%.10f\n", truth.beta)
        @printf(io, "sigma=%.10f\n", truth.sigma)
        for j in 1:J
            @printf(io, "alpha_%d=%.10f\n", j, alpha[j])
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    J = parse(Int, ARGS[1])
    n_obs = parse(Int, ARGS[2])
    seed = parse(Int, ARGS[3])
    data_dir = ARGS[4]
    write_dataset(data_dir, J, n_obs, seed)
    println("Wrote dataset to $data_dir (J=$J, n_obs=$n_obs, seed=$seed)")
end
