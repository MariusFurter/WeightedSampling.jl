# run_ws.jl
#
# Loads a dataset written by `../simulate.jl`, runs the hierarchical regression
# `@model` (`model.jl`) with `N` particles, and prints timing + inference-quality
# metrics as `key=value` lines to stdout.
#
# Usage: `julia --project=. run_ws.jl <data_dir> <N> [seed]`
using Random
using Printf
include(joinpath(@__DIR__, "model.jl"))

function read_data(data_dir::AbstractString)
    lines = readlines(joinpath(data_dir, "data.csv"))
    J = 0
    rows = Tuple{Int,Float64,Float64}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line, ',')
        g = parse(Int, parts[1])
        x = parse(Float64, parts[2])
        y = parse(Float64, parts[3])
        push!(rows, (g, x, y))
        J = max(J, g)
    end
    groups = [Tuple{Float64,Float64}[] for _ in 1:J]
    for (g, x, y) in rows
        push!(groups[g], (x, y))
    end
    return J, groups
end

function read_true_params(data_dir::AbstractString)
    truth = Dict{String,Float64}()
    for line in readlines(joinpath(data_dir, "true_params.txt"))
        isempty(strip(line)) && continue
        k, v = split(line, '=')
        truth[k] = parse(Float64, v)
    end
    return truth
end

function run_ws_benchmark(data_dir::AbstractString, N::Int; seed::Int=1)
    J, groups = read_data(data_dir)
    truth = read_true_params(data_dir)
    true_alpha = [truth["alpha_$j"] for j in 1:J]

    Random.seed!(seed)

    # Warm-up run (tiny, same argument types) to exclude JIT compilation from
    # the timed run.
    warmup_model = hierarchical_regression(1, [[(0.0, 0.0)]])
    warmup_state = SMCState(50)
    run!(warmup_model, warmup_state)

    model = hierarchical_regression(J, groups)
    state = SMCState(N)

    stats = @timed run!(model, state)
    elapsed = stats.time - stats.compile_time

    weights = state.weights
    alpha_samples = state[:alpha]
    alpha_est = [expectation([p[j] for p in alpha_samples], weights) for j in 1:J]
    mu_alpha_est = expectation(state[:mu_alpha], weights)
    tau_alpha_est = expectation(state[:tau_alpha], weights)
    beta_est = expectation(state[:beta], weights)
    sigma_est = expectation(state[:sigma], weights)

    rmse_alpha = sqrt(sum((alpha_est .- true_alpha) .^ 2) / J)

    # Particle effective sample size: ESS = 1 / sum(w_i^2) for normalized
    # weights `w` (standard SMC effective sample size, comparable to an MCMC
    # ESS: "how many independent draws is this weighted particle set worth").
    ess = N * WeightedSampling.ess_perc(exp_norm(weights))

    @printf("J=%d\n", J)
    @printf("n_obs=%d\n", length(groups[1]))
    @printf("N=%d\n", N)
    @printf("elapsed_time=%.6f\n", elapsed)
    @printf("alloc_mib=%.4f\n", stats.bytes / 2^20)
    @printf("ess=%.4f\n", ess)
    @printf("time_per_ess=%.6f\n", elapsed / ess)
    @printf("rmse_alpha=%.6f\n", rmse_alpha)
    @printf("mu_alpha_est=%.6f\n", mu_alpha_est)
    @printf("mu_alpha_err=%.6f\n", abs(mu_alpha_est - truth["mu_alpha"]))
    @printf("tau_alpha_est=%.6f\n", tau_alpha_est)
    @printf("tau_alpha_err=%.6f\n", abs(tau_alpha_est - truth["tau_alpha"]))
    @printf("beta_est=%.6f\n", beta_est)
    @printf("beta_err=%.6f\n", abs(beta_est - truth["beta"]))
    @printf("sigma_est=%.6f\n", sigma_est)
    @printf("sigma_err=%.6f\n", abs(sigma_est - truth["sigma"]))

    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = ARGS[1]
    N = parse(Int, ARGS[2])
    seed = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    run_ws_benchmark(data_dir, N; seed=seed)
end
