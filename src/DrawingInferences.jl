module DrawingInferences

export WeightedSampler, FKStep, FKModel, variables, @fk, SMCModel, SMCIO, smc!

include("smc.jl")
include("dsl.jl")

end
