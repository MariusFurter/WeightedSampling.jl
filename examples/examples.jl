using Revise
using DrawingInferences
using DataFrames
using Distributions
using BenchmarkTools

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

@smc function bla()
    x0 = 0.0
    for i in 1:1_000
        x{i} ~ walkKernel(i)
        i => walkKernel(x{i})
    end
end

my_kernels = (initialKernel=initialKernel, walkKernel=walkKernel)

my_particles, evidence = @time bla(n_particles=1_000, kernels=my_kernels)
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