"""
Compute inverse CDF induced by the normalized `weights``. `us`` must be a sorted vector of uniform random samples.
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

function stratified_resample(weights::Vector{Float64})
    N = length(weights)

    # Generate uniform random samples us[n] ~ U([(n-1)/N, n/N))
    us = Vector{Float64}(undef, N)
    inv_N = 1.0 / N
    for n in 1:N
        @inbounds us[n] = (n - 1) * inv_N + rand() * inv_N
    end

    # Compute indices using the inverse CDF
    indices = icdf(weights, us)
    return indices
end

"""
Compute effective sample size percentage (ESS/N) from weights. 
"""
function ess_perc(weights::Vector{Float64})
    N = length(weights)
    return 1.0 / (N * sum(weights .^ 2))
end


function resample_particles!(particles, ess_perc_min=0.5::Float64)
    exp_norm_weights!(particles[!, :weights])
    if ess_perc(particles[!, :weights]) < ess_perc_min
        indices = stratified_resample(particles[!, :weights])

        for col in names(particles)
            particles[!, col] = particles[indices, col]
        end
        particles[!, :weights] .= 0.0
    else
        particles[!, :weights] .= log.(particles[!, :weights])
    end
    return nothing
end
