struct Sampler{S,L}
    sampler::S
    logpdf::L
end

struct WeightedSampler{S,L,W}
    sampler::S # (input_args, rng) -> sampled_value
    logpdf::L # (input_args, sampled_value) -> log probability
    weighter::W # (input_args, sampled_value) -> weight
end

function fromDistribution(d)
    return WeightedSampler(
        (rng) -> rand(rng, d),
        (x) -> logpdf(d, x),
        (x) -> 0.0
    )
end

function importanceSampler(proposal, target)
    return WeightedSampler(
        (x, rng) -> rand(rng, proposal),
        (x) -> logpdf(proposal, x),
        (x) -> target(x) - logpdf(proposal, x)
    )
end