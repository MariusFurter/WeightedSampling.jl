# MH Linear Regression Benchmark
# Linear regression with MH random walk moves on every observation step.
# Ported from WeightedSampling.torch benchmarks/mh_linear_regression.py
#
# Usage: julia --project benchmarks/mh_linear_regression.jl

using WeightedSampling
using Distributions
using Random
using Printf

function generate_synthetic_data(num_points=20)
    Random.seed!(0)
    xs = range(0, 10, length=num_points)
    true_a = 2.0
    true_b = -1.0
    true_sigma = 0.5
    ys = true_a .* xs .+ true_b .+ true_sigma .* randn(num_points)
    return collect(zip(xs, ys))
end

@smc function linear_model(data)
    a ~ Normal(0.0, 5.0)
    b ~ Normal(0.0, 5.0)
    for (x, y) in data
        y => Normal(a * x + b, 0.5)
        (a, b) << RW(0.1)
    end
end

function benchmark_mh()
    num_particles = 1_000
    num_points = 50
    data = generate_synthetic_data(num_points)

    println("Running MH Linear Regression Benchmark...")
    @printf("Particles: %d\n", num_particles)
    @printf("Data Points: %d\n", num_points)
    @printf("Total Moves: %d\n", num_points)

    # Warmup run
    linear_model(data; n_particles=100, show_progress=false)

    # Timed run
    t = @elapsed begin
        particles, evidence = linear_model(data; n_particles=num_particles, show_progress=false)
    end

    @printf("Done.\n")
    @printf("Total Time: %.4fs\n", t)
    @printf("Moves/sec: %.2f\n", num_points / t)
    @printf("Log Evidence: %.4f\n", evidence)
end

benchmark_mh()
