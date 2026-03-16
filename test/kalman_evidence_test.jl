### Kalman filter log-evidence test
# Linear-Gaussian SSM:
#   x_0 ~ Normal(0, 1)
#   x_t = 0.8 * x_{t-1} + Normal(0, 0.5)
#   y_t ~ Normal(x_t, 0.5)
#
# Compare SMC log-evidence against exact Kalman filter solution.
# Ported from WeightedSampling.torch examples/verify_log_evidence.py

function kalman_filter_evidence(data)
    F = 0.8       # transition coefficient
    Q = 0.5^2     # process noise variance
    H = 1.0       # observation coefficient
    R = 0.5^2     # observation noise variance

    # Initial belief: x_0 ~ N(0, 1)
    μ = 0.0
    P = 1.0
    log_evidence = 0.0

    for y in data
        # Predict
        μ_pred = F * μ
        P_pred = F^2 * P + Q

        # Innovation
        S = H^2 * P_pred + R
        residual = y - H * μ_pred
        log_evidence += -0.5 * (log(2π) + log(S) + residual^2 / S)

        # Update
        K = P_pred * H / S
        μ = μ_pred + K * residual
        P = (1 - K * H) * P_pred
    end

    return log_evidence
end

function kalman_evidence_test(; n_particles=10_000, T=50, max_abs_diff=3.0)
    Random.seed!(42)

    # Generate synthetic data from the linear-Gaussian SSM
    x_prev = randn()
    observations = Float64[]
    for _ in 1:T
        x_curr = 0.8 * x_prev + 0.5 * randn()
        push!(observations, x_curr + 0.5 * randn())
        x_prev = x_curr
    end

    # Exact solution
    exact = kalman_filter_evidence(observations)

    # SMC estimate
    @smc function ssm_model(data)
        x ~ Normal(0.0, 1.0)
        for (t, y) in enumerate(data)
            x ~ Normal(0.8 * x, 0.5)
            y => Normal(x, 0.5)
        end
    end

    particles, smc_evidence = ssm_model(observations; n_particles=n_particles, show_progress=false)

    diff = abs(exact - smc_evidence)
    return diff < max_abs_diff
end
