"""
    RW(particles, targets, step_size)

Symmetric random walk proposal kernel with variance `step_size`.

# Arguments
- `particles::DataFrame`: Current particle set
- `targets::Vector{Symbol}`: Variable names to update
- `step_size`: Step size for random walk

# Returns
- `proposals::DataFrame`: Proposed new values for target variables
- `log_ratios::Vector{Float64}`: Log proposal ratios (zero since proposal is symmetric)

# Examples  
```julia
@smc function model()
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)
    
    α << RW(0.1)
    (α, β) << RW(0.1)
end
```

See also: [`@smc`](@ref)
"""
function RW(particles, targets, step_size)
    # Gaussian RW independent Gaussian steps of step_size
    N = nrow(particles)
    d = length(targets)

    # Cover case where step_size is an array
    if step_size isa AbstractArray && !isempty(step_size)
        step_size = step_size[1]
    end

    changes = rand(MvNormal(zeros(d), step_size), N)

    df = DataFrame()
    for (i, col) in enumerate(targets)
        df[!, col] = changes[i, :] .+ particles[!, col]
    end

    return df, zeros(N)
end

"""
    autoRW(particles, targets, min_step=1e-3)

Adaptive random walk proposal kernel with empirically calibrated covariance.

# Arguments
- `particles::DataFrame`: Current particle set  
- `targets::Vector{Symbol}`: Variable names to update
- `min_step=1e-3`: Minimum step size to prevent singular covariance

# Returns
- `proposals::DataFrame`: Proposed new values for target variables
- `log_ratios::Vector{Float64}`: Log proposal ratios (zero since proposal is symmetric)

# Description
Performs adaptive random walk using the empirical covariance of target particles:
```math
x^{\\text{new}} = x^{\\text{old}} + \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, \\lambda \\Sigma)
```
where:
- `` \\lambda = 2.38 d^(-1/2) `` is the optimal scaling factor for ``d``-dimensional problems
- ``\\Sigma`` is the weighted empirical covariance matrix of the target particles
- Covariance elements are replaced with `min_step` if they are zero

# Examples  
```julia
@smc function model()
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)
    
    α << autoRW()
    (α, β) << autoRW()
end
```

See also: [`@smc`](@ref)
"""
function autoRW(particles, targets, min_step=1e-3)
    # Gaussian RW with covariance λΣ
    # where λ = 2.38 d^-1/2 and Σ is the empirical covariance matrix of the target particles
    # targets :: Vector
    N = nrow(particles)

    # Cover case where min_step is an array
    if min_step isa AbstractArray && !isempty(min_step)
        min_step = min_step[1]
    end

    d = length(targets)
    λ = 2.38 * d^(-1 / 2)

    m = Matrix(particles[!, targets])
    w = ProbabilityWeights(exp_norm(particles[!, :weights]))
    Σ = cov(m, w)

    # Replace 0.0 values with minimum step epsilon
    Σ[Σ.==0.0] .= min_step

    changes = rand(MvNormal(λ * Σ), N)

    df = DataFrame()
    for (i, col) in enumerate(targets)
        df[!, col] = changes[i, :] .+ particles[!, col]
    end

    return df, zeros(N)
end

default_proposals = (
    RW=RW,
    autoRW=autoRW,
)
