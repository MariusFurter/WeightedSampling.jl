module WeightedSampling

using DataFrames
using Distributions
using MacroTools
using ProgressMeter
using StaticArrays
using StatsBase

export SMCKernel, @smc, @E, exp_norm, diversity, sample_particles, describe_particles, RW, autoRW

include("rewrites.jl")
include("resampling.jl")
include("moves.jl")
include("smc_kernels.jl")
include("move_kernels.jl")
include("utils.jl")

end
