using Distributions

struct WeightedSampler{S,W,O}
    sampler::S
    weighter::W
    output_type::O
end

function fromDistribution(d::Distribution)
    return WeightedSampler(
        (rng) -> rand(rng, d),
        () -> 0.0,
        () -> eltype(d)
    )
end

function importanceSampler(proposal::Distribution, target)
    return WeightedSampler(
        (x, rng) -> rand(rng, proposal),
        (x) -> target(x) - logpdf(proposal, x),
        () -> eltype(proposal)
    )
end