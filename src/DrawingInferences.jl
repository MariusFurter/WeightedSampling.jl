module DrawingInferences

using DataFrames
using Distributions
using MacroTools

export SMCKernel, @smc, @E

include("rewrites.jl")
include("resampling.jl")
include("smc_kernels.jl")
include("utils.jl")

end
