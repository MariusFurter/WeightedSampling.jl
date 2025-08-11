using Revise
using DrawingInferences
using DataFrames
using Distributions

@smc function normal_model()
    x ~ Normal(0, 1)
    y ~ Normal(x, 1)
    z ~ Normal(y, 1)
    u = sin(x)
end

samples = normal_model(n_particles=10_000, ess_perc_min=0.5)
s = 1
@E(x + s, samples)

initialKernel = SMCKernel(
    () -> randn(),
    (x) -> logpdf(Normal(0, 1), x),
    nothing
)

walkKernel = SMCKernel(
    (x_in::Float64) -> x_in + randn(),
    (x_in::Float64, x_out::Float64) -> logpdf(Normal(x_in, 1), x_out),
    nothing
)

@smc function bla()
    x0 = 0.0
    for i in 1:10_000
        x{i} = x{i - 1} + i
        x{i} ~ walkKernel(x{i - 1})
    end
    0.0 -> walkKernel(x1)
end

my_kernels = (initialKernel=initialKernel, walkKernel=walkKernel)
my_particles = DataFrame(weights=[0.0 for _ in 1:10_000])

@time bla!(my_particles, my_kernels, 1.0)
describe(my_particles)


### Resampling Benchmarking
particles_template = DataFrame(x=[n for n in 1:1000], weights=randn(1000))

@btime DrawingInferences.exp_norm_weights!(particles.weights) setup = (particles = copy(particles_template)) # 4us, 0 allocs
#No resampling needed
@btime DrawingInferences.resample_particles!(particles, 0.0) setup = (particles = copy(particles_template)) # 10us, 16 allocs
#Resampling needed
@btime DrawingInferences.resample_particles!(particles, 1.0) setup = (particles = copy(particles_template)) # 16us, 35 allocs
particles