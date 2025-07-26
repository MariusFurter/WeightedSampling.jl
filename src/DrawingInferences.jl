module DrawingInferences

export WeightedSampler, FKStep, FKModel, variables, @model, SMCModel, SMCIO, smc!

include("samplers.jl")
include("smc.jl")
include("dsl.jl")

end
