"""
Resampling machinery: inverse-CDF stratified resampling, ESS, and
log-sum-exp helpers used by `Resample` and evidence computation.
"""

"""
    icdf(weights::Vector{Float64}, us::Vector{Float64})

Inverse-CDF lookup: for each (sorted) uniform sample in `us`, find the index
`m` such that the cumulative sum of `weights` up to `m` first exceeds it.
`weights` must be normalized (sum to 1) and `us` sorted ascending.
"""
function icdf(weights::Vector{Float64}, us::Vector{Float64})
    N = length(weights)
    indices = zeros(Int, N)
    s = weights[1]
    m = 1
    for n in 1:N
        while @inbounds s < us[n]
            m += 1
            s += weights[m]
        end
        @inbounds indices[n] = m
    end
    return indices
end

"""
    stratified_resample(weights::Vector{Float64})

Stratified resampling: draws `N` indices with (normalized) probability
`weights[i]`, using one stratified uniform sample per output slot (lower
variance than plain multinomial resampling).
"""
function stratified_resample(weights::Vector{Float64})
    N = length(weights)
    us = Vector{Float64}(undef, N)
    inv_N = 1.0 / N
    for n in 1:N
        @inbounds us[n] = (n - 1) * inv_N + rand() * inv_N
    end
    return icdf(weights, us)
end

"""
    ess_perc(weights::Vector{Float64})

Effective sample size, as a percentage of `N`: `ESS / N = 1 / (N * sum(w^2))`
for normalized weights `w`.
"""
function ess_perc(weights::Vector{Float64})
    N = length(weights)
    return 1.0 / (N * sum(abs2, weights))
end

"""
    logsumexp(logw)

`log(sum(exp.(logw)))`, computed in a numerically stable way.
"""
function logsumexp(logw)
    m = maximum(logw)
    return m + log(sum(x -> exp(x - m), logw))
end

"""
    exp_norm(weights::Vector{Float64})

Exponentiate and normalize (unnormalized) log-`weights` into a probability
vector, subtracting the max first for numerical stability.
"""
function exp_norm(weights::Vector{Float64})
    m = maximum(weights)
    w = exp.(weights .- m)
    w ./= sum(w)
    return w
end
