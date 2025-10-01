using Revise
using WeightedSampling
using DataFrames
using Distributions
using BenchmarkTools

using LinearAlgebra
using StaticArrays

df = DataFrame(x=fill(SVector{1000,Float64}(zeros(1000)), 1000), weights=randn(1000))

indices = sort(rand(1:1000, 1000))


function copytest(df, indices)
    df[:, :] = df[indices, :]
    for col in names(df)
        if eltype(df[!, col]) <: AbstractArray
            df[!, col] .= deepcopy.(df[!, col])
        end
    end
end

@time copytest(df, indices)

@smc function test_array()
    x = [0.0, 0.0]
    for i in 1:100
        y ~ MvNormal(SVector(x[1], 0.0), Diagonal([1.0, 1.0]))
        z = [x[1], y[1], x[2], y[2]]
    end
    z[1] ~ Normal(x[2], 0.001)
end

samples, evidence = @time test_array(n_particles=10_000, ess_perc_min=0.5)

@smc function test(T)
    x .= Vector{Float64}(undef, T)
    for i in 1:T
        x[i] ~ Normal(0, 1)
        0 => Normal(x[i], 1)
    end
end

samples, evidence = @time test(1_000; n_particles=1_000, ess_perc_min=0.5)
# (copy) 1k x 1k: 0.4s, 0.66M allocs
# (copy with static arrays) 1k x 1k: 0.5s, 260k allocs
# (deepcopy) 1k x 1k: 0.8s, 1.24M allocs
# (deepcopy with static arrays) 1k x 1k: 0.4s, 853k allocs
#10k x 1k: 26s, 6.57M allocs

unique(map(x -> x[1000], samples.x))

@smc function test2(T)
    for i in 1:T
        x{i} ~ Normal(0, 1)
        0 => Normal(x{i}, 1)
    end
end

samples, evidence = @time test2(1_000; n_particles=10_000, ess_perc_min=0.5)
#10k x 1k: 34s, 60M allocs

@smc function normal_model()
    x ~ Normal(0, 1)
    y ~ Normal(x, 1)
    3 => Normal(y, 1)
    u .= sin(x)
end

@time samples, evidence = normal_model(n_particles=10_000, ess_perc_min=0.5)
@E((x, y) -> x, samples)

### Dynamic-style model

initialKernel = SMCKernel(
    () -> randn(),
    (x) -> logpdf(Normal(0, 1), x),
    nothing
)

walkKernel = SMCKernel(
    (x_in) -> x_in + randn(),
    (x_in, x_out) -> logpdf(Normal(x_in, 1), x_out),
    nothing
)

@smc function walk()
    x0 .= 0.0
    for i in 1:10_000
        x{i} ~ walkKernel(i)
        i => walkKernel(x{i})
    end
end

my_kernels = (initialKernel=initialKernel, walkKernel=walkKernel)

my_particles, evidence = @time walk(n_particles=1_000, kernels=my_kernels)
describe(my_particles)

@smc function walk_v()
    x .= Vector{Float64}(undef, 10_000)
    x[1] .= 0.0
    for i in 2:10_000
        x[i] ~ walkKernel(i)
        i => walkKernel(x[i])
    end
end

my_particles, evidence = @time walk_v(n_particles=1_000, kernels=my_kernels)

my_particles.x[1]

describe(my_particles)

### Beta-Binomial Model Benchmarking

n = 10
k = 8
T = 10
a = 1.0
b = 2.0

@smc function beta_binomial(n, k, T, a, b)
    a = a
    b = b
    p ~ Beta(a, b)
    for t in 1:T
        k => Binomial(n, p)
    end
end

samples, evidence = @time beta_binomial(n, k, T, a, b; n_particles=10_000, ess_perc_min=1.0)

smc_p_val = @E(p -> p, samples)


### Resampling Benchmarking
particles_template = DataFrame(x=[n for n in 1:1000], weights=randn(1000))

@btime WeightedSampling.exp_norm_weights!(particles.weights) setup = (particles = copy(particles_template)) # 4us, 0 allocs
#No resampling needed
@btime WeightedSampling.resample_particles!(particles, 0.0) setup = (particles = copy(particles_template)) # 10us, 16 allocs
#Resampling needed
@btime WeightedSampling.resample_particles!(particles, 1.0) setup = (particles = copy(particles_template)) # 16us, 35 allocs
particles