# MH Linear Regression Benchmark — WeightedSampling.jl (SMC)
#
# Linear regression with MH random walk moves on every observation step.
#
# Usage: julia --project benchmarks/mh_linear_regression.jl

using WeightedSampling
using Distributions
using Random
using Printf
using Statistics

function generate_synthetic_data(num_points)
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
    n_runs = 5
    data = generate_synthetic_data(num_points)

    println("="^60)
    println("MH Linear Regression — WeightedSampling.jl (SMC)")
    println("="^60)
    @printf("Particles: %d\n", num_particles)
    @printf("Data points: %d\n", num_points)
    @printf("Runs: %d (median reported)\n", n_runs)

    # Warmup (compilation)
    linear_model(data; n_particles=100, show_progress=false)

    times = Float64[]
    local particles, evidence
    for run in 1:n_runs
        Random.seed!(42 + run)
        t = @elapsed begin
            particles, evidence = linear_model(data;
                n_particles=num_particles, show_progress=false)
        end
        push!(times, t)
    end

    med_time = median(times)

    a_est = @E(a -> a, particles)
    b_est = @E(b -> b, particles)

    @printf("Median time:  %.4f s  (range: %.4f – %.4f)\n",
        med_time, minimum(times), maximum(times))
    @printf("Moves/sec:    %.0f\n", num_points / med_time)
    @printf("Log evidence: %.4f\n", evidence)
    @printf("a:            %.4f  (true: 2.0)\n", a_est)
    @printf("b:            %.4f  (true: -1.0)\n", b_est)
end

benchmark_mh()
