module WeightedSampling

using DataFrames
using Distributions
using LinearAlgebra: I
using MacroTools
using ProgressMeter
using Random
using StatsBase

export
    # Core types
    SMCState, WeightedKernel,
    run!,

    # MH proposal kernels
    exp_norm, RW, autoRW, default_proposals,

    # Default sampling/observation kernels
    default_kernels,

    # Particle analysis utilities
    expectation, @E, sample, describe, log_evidence,

    # Macro-based model construction
    @model

include("stores.jl")
include("types.jl")
include("resampling.jl")
include("transformers.jl")
include("move_kernels.jl")
include("default_kernels.jl")
include("utils.jl")
include("rewrites.jl")

end
