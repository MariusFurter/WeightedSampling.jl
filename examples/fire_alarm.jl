# A Bayesian network for a fire alarm 

using WeightedSampling
using Random
using CairoMakie

Random.seed!(42)

@model function fire_alarm()
    fire ~ Bernoulli(0.01)
    smoke ~ Bernoulli(fire ? 0.9 : 0.01)
    lever ~ Bernoulli(fire ? 0.7 : 0.01)
    alarm ~ Bernoulli(smoke || lever ? 0.98 : 0.01)
end

state = SMCState(100_000)
model = fire_alarm()
run!(model, state)
describe(state)

# Probability of fire without smoke
@E((fire,smoke) -> fire & !smoke, state)

# Conditioning on alarm=true #
@model function fire_alarm()
    fire ~ Bernoulli(0.01)
    smoke ~ Bernoulli(fire ? 0.9 : 0.01)
    lever ~ Bernoulli(fire ? 0.7 : 0.01)
    true => Bernoulli(smoke || lever ? 0.98 : 0.01)
end

state = SMCState(100_000)
model = fire_alarm()
run!(model, state)
describe(state)