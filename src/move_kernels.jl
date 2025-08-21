## Proposal kernel should take particles and targets as input
## and return a df of suggested changes and a vector of log scores representing q(x_old | x_new)/q(x_new | x_old).

function RW(particles, targets, step_size)
    # Gaussian RW independent Gaussian steps of step_size
    N = nrow(particles)
    targets = collect(targets)
    d = length(targets)

    # Cover case where step_size is an array
    if step_size isa AbstractArray && !isempty(step_size)
        step_size = step_size[1]
    end

    changes = rand(MvNormal(zeros(d), step_size), N)

    for (i, col) in enumerate(targets)
        changes[i, :] .+= particles[!, col]
    end

    return DataFrame(changes', targets), zeros(N)
end

function autoRW(particles, targets, min_step=1e-3)
    # Gaussian RW with covariance λΣ
    # where λ = 2.38 d^-1/2 and Σ is the empirical covariance matrix of the target particles
    # targets :: Vector
    N = nrow(particles)
    targets = collect(targets)

    # Cover case where min_step is an array
    if min_step isa AbstractArray && !isempty(min_step)
        min_step = min_step[1]
    end

    d = length(targets)
    λ = 2.38 * d^(-1 / 2)

    m = Matrix(particles[!, targets])
    w = ProbabilityWeights(exp_norm_weights(particles[!, :weights]))
    Σ = cov(m, w)

    # Replace values with minimum step epsilon
    Σ[Σ.<=min_step] .= min_step

    changes = rand(MvNormal(λ * Σ), N)

    for (i, col) in enumerate(targets)
        changes[i, :] .+= particles[!, col]
    end

    return DataFrame(changes', targets), zeros(N)
end

default_proposals = (
    RW=RW,
    autoRW=autoRW,
)
