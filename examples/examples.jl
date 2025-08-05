using Revise
using DrawingInferences
using DataFrames
using Distributions

initialKernel = SMCKernel(
    () -> randn(),
    (x) -> logpdf(Normal(0, 1), x),
    nothing
)

walkKernel = SMCKernel(
    (x_in::Float64) -> x_in + randn(),
    (x_in::Float64, x_out::Float64) -> logpdf(Normal(x_in, 1), x_out),
    (x_in::Float64, x_out::Float64) -> 0.0
)

my_kernels = (initialKernel=initialKernel, walkKernel=walkKernel)

my_particles = DataFrame(weights=[0.0 for _ in 1:1000])

@smc function bla()
    x0 = 0.0
    for i in 1:1000
        x{i} = x{i - 1} + i
        x{i} ~ walkKernel(x{i - 1})
    end
    0.0 -> walkKernel(x1)
end

@time bla!(my_particles, my_kernels)
describe(my_particles)