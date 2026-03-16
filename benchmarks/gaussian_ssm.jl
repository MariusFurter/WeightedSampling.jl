# Gaussian SSM Benchmark
# Bootstrap particle filter for a 1D random walk state-space model.
# Ported from WeightedSampling.torch benchmarks/gaussian_ssm.py
#
# Usage: julia --project benchmarks/gaussian_ssm.jl

using WeightedSampling
using Distributions
using Random
using Printf

function generate_data(num_timesteps)
    Random.seed!(42)
    x_curr = 0.0
    y = Float64[]
    for t in 1:num_timesteps
        x_curr = x_curr + randn()
        push!(y, x_curr + randn())
    end
    return y
end

@smc function ssm_model(data)
    x ~ Normal(0.0, 1.0)
    for (t, y) in enumerate(data)
        x ~ Normal(x, 1.0)
        y => Normal(x, 1.0)
    end
end

function benchmark_ssm()
    num_particles = 10_000
    num_timesteps = 100
    data = generate_data(num_timesteps)

    println("Running Gaussian SSM Benchmark...")
    @printf("Particles: %d\n", num_particles)
    @printf("Timesteps: %d\n", num_timesteps)

    # Warmup run
    ssm_model(data; n_particles=100, show_progress=false)

    # Timed run
    t = @elapsed begin
        particles, evidence = ssm_model(data; n_particles=num_particles, show_progress=false)
    end

    @printf("Done.\n")
    @printf("Total Time: %.4fs\n", t)
    @printf("Steps/sec: %.2f\n", num_timesteps / t)
    @printf("Log Evidence: %.4f\n", evidence)
end

benchmark_ssm()
