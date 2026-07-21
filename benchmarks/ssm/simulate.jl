# Shared data-generation + exact-reference helpers for the `benchmarks/ssm/`
# 1D linear-Gaussian state-space model benchmarks (WeightedSampling and
# SequentialMonteCarlo.jl). Kept dependency-free (no CSV.jl/DataFrames) since
# each benchmark subfolder has its own Project.toml environment.
#
#     x(0) ~ Normal(0, x0_std)
#     x(t) = a * x(t-1) + w(t),   w(t) ~ Normal(0, q)
#     y(t) ~ Normal(x(t), r)
#
# libbi (`benchmarks/ssm/libbi/lgssm1d/`) is NOT driven by this file — it
# generates its own data via `libbi sample --target joint` with a matching
# T/seed (statistically equivalent, not bit-identical; see repo memory notes).

"""
    simulate_lgssm1d(T, a, q, r, x0_std)

Simulate one trajectory/observation sequence from the model above, returning
`(states, observations)` of length `T`. Uses the currently active global RNG
(caller should `Random.seed!(...)` beforehand for reproducibility).
"""
function simulate_lgssm1d(T::Int, a, q, r, x0_std)
    states = Vector{Float64}(undef, T)
    observations = Vector{Float64}(undef, T)
    x = x0_std * randn()
    for t in 1:T
        x = a * x + q * randn()
        states[t] = x
        observations[t] = x + r * randn()
    end
    return states, observations
end

"""
    kalman_filter_evidence(data, a, q, r, x0_std)

Exact Kalman filter for this model. Returns `(posterior_mean_final,
log_evidence)`, used as a correctness check for the particle filters.
"""
function kalman_filter_evidence(data, a, q, r, x0_std)
    μ, P = 0.0, x0_std^2
    log_evidence = 0.0
    for y in data
        μ_pred = a * μ
        P_pred = a^2 * P + q^2

        S = P_pred + r^2
        residual = y - μ_pred
        log_evidence += -0.5 * (log(2π) + log(S) + residual^2 / S)

        K = P_pred / S
        μ = μ_pred + K * residual
        P = (1 - K) * P_pred
    end
    return μ, log_evidence
end

"""
    write_data_csv(path, states, observations)

Write `t,x,y` rows to `path` (creating parent directories as needed), for
optional manual inspection/reuse. Not required by any of the benchmark
scripts themselves.
"""
function write_data_csv(path::AbstractString, states::Vector{Float64}, observations::Vector{Float64})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "t,x,y")
        for t in eachindex(states)
            println(io, t, ",", states[t], ",", observations[t])
        end
    end
    return path
end
