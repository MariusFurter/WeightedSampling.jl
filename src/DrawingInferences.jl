module DrawingInferences

export WeightedSampler, FKStep, FKModel, variables, @fkstep, @fk, SMCModel, SMCIO, smc!

include("smc.jl")
include("dsl.jl")

end
