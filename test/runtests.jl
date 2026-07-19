using Test
using WeightedSampling
using WeightedSampling: apply!, score!, score_logpdf, score_logpdf!,
    AbstractParticleStore, ColumnStore,
    nparticles, hascol, getcol, colnames, broadcast_setcol!, resample!,
    ParticleTransformer, ScoreCtx,
    Assign, AccessorAssign, AccessorSample, Sample, Observe, Weight,
    Sequence, Loop, Cond, Resample, Move,
    icdf, stratified_resample, ess_perc, logsumexp,
    dynname

include("models.jl")

@testset "WeightedSampling.jl" begin
    include("transformers_test.jl")
    include("score_test.jl")
    include("move_test.jl")
    include("macro_test.jl")
    include("move_macro_test.jl")
    include("accessors_test.jl")
    include("dynamic_vars_test.jl")
    include("dynamic_move_test.jl")
    include("default_kernels_test.jl")
    include("api_test.jl")
    include("show_test.jl")
end
