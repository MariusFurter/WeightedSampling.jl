struct Sampler{S,L,O}
    sampler::S
    logpdf::L
    output_type::O
end

struct WeightedSampler{S,L,W,O}
    sampler::S
    logpdf::L
    weighter::W
    output_type::O
end

function fromSampler(s::Sampler)
    return WeightedSampler(
        s.sampler,
        s.logpdf,
        () -> 0.0,
        s.output_type
    )
end

function fromDistribution(d)
    return WeightedSampler(
        (rng) -> rand(rng, d),
        () -> (x) -> logpdf(d, x),
        () -> 0.0,
        () -> eltype(d)
    )
end

function importanceSampler(proposal, target)
    return WeightedSampler(
        (x, rng) -> rand(rng, proposal),
        () -> (x) -> logpdf(proposal, x),
        (x) -> target(x) - logpdf(proposal, x),
        () -> eltype(proposal)
    )
end