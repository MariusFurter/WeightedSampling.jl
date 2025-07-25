module DrawingInferences

export WeightedSampler, FKStep, FKModel, variables, @fkstep, @fk, SMCModel, SMCIO, smc!

include("sampler.jl")
include("smc.jl")
include("dsl.jl")

end
