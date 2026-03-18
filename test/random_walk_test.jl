### Sequential random walk test
# 1D random walk state-space model:
#   x_0 ~ Normal(0, 1)
#   x_{t+1} ~ Normal(x_t, 1)
#   y_t ~ Normal(x_{t+1}, 0.5)
#
# Ported from WeightedSampling.torch test_random_walk.py

function random_walk_test(; n_particles=10_000, T=10, max_error=1.5)
    Random.seed!(42)

    # Generate synthetic data
    true_x = [0.0]
    data = Float64[]
    for t in 1:T
        next_val = true_x[end] + randn()
        push!(true_x, next_val)
        obs = next_val + 0.5 * randn()
        push!(data, obs)
    end

    @model function random_walk_model(data)
        x ~ Normal(0.0, 1.0)
        for (t, y) in enumerate(data)
            x ~ Normal(x, 1.0)
            y => Normal(x, 0.5)
        end
    end

    particles, evidence = random_walk_model(data; n_particles=n_particles, show_progress=false)

    estimated_mean = @E(x -> x, particles)
    true_last = true_x[end]

    error = abs(estimated_mean - true_last)
    return error < max_error
end
