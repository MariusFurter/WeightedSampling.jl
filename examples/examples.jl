using Revise
using DrawingInferences
using DataFrames
using Distributions
using BenchmarkTools

#Issues with broadcasting -> solution is to fill(-,N) all literal values. Think about how to recognize these better.
# - Also, how to fill in arrays that references variables?

using LinearAlgebra
using StaticArrays

df = DataFrame(x=randn(1000), y=randn(1000), z=randn(1000))

function cat(df)
    broadcast(vcat, df.x, df.y, df.x, df.z)
end
@time cat(df)


function cat2(df)
    broadcast(SVector, df.x, df.y, df.x, df.z)
end
@time cat2(df)

@smc function test()
    #x = [0.0, 0.0]
    x = zeros(1000)
    for i in 1:1000
        x[i] ~ Normal(0, 1)
    end
    #y ~ MvNormal([x[1], x[2]], Diagonal([1.0, 1.0]))
    #z = [y[1], 3.0]
end

samples, evidence = @time test()

@smc function normal_model()
    x ~ Normal(0, 1)
    y ~ Normal(x, 1)
    3 => Normal(y, 1)
    u = sin(x)
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
    x0 = 0.0
    for i in 1:10_000
        x{i} ~ walkKernel(i)
        i => walkKernel(x{i})
    end
end

my_kernels = (initialKernel=initialKernel, walkKernel=walkKernel)

my_particles, evidence = @time walk(n_particles=1_000, kernels=my_kernels)
describe(my_particles)

@smc function walk_v()
    x = Vector{Float64}(undef, 10_000)
    x[1] = 0.0
    for i in 2:10_000
        x[i] ~ walkKernel(i)
        i => walkKernel(x[i])
    end
end

my_particles, evidence = @time walk_v(n_particles=1_000, kernels=my_kernels)

## Solves the wide df issue!
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

@btime DrawingInferences.exp_norm_weights!(particles.weights) setup = (particles = copy(particles_template)) # 4us, 0 allocs
#No resampling needed
@btime DrawingInferences.resample_particles!(particles, 0.0) setup = (particles = copy(particles_template)) # 10us, 16 allocs
#Resampling needed
@btime DrawingInferences.resample_particles!(particles, 1.0) setup = (particles = copy(particles_template)) # 16us, 35 allocs
particles