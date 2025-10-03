`WeightedSampling.jl` provides the `@smc` macro for concise specification of Sequential Monte Carlo (SMC) sampling schemes. It transforms probabilistic programs into efficient particle filters with automatic resampling and weight management. It excels at dynamic and moderate-dimensional statistical models (< 1k parameters).

**Note:** This package is in early development. Use with caution.

## Installation

The package is not yet registered in the Julia General Registry. Install directly from this repository:

```terminal
add https://github.com/MariusFurter/WeightedSampling.jl
```

## Quick start

### Linear regression

```julia
using WeightedSampling

@smc function linear_regression(xs, ys)
    α ~ Normal(0, 1) # sample
    β ~ Normal(0, 1)
    for (x, y) in zip(xs, ys)
        y => Normal(α + β * x, 1.0) # observe
    end
end

xs = 1:10
ys = 1-0 .- 0.5 .* xs .+ 0.5 * randn(length(xs))

particles, evidence = linear_regression(xs, ys, n_particles=1000, ess_perc_min=0.5)
describe_particles(particles)
```

### Bootstrap particle filter

```julia
@smc function ssm(obs)
    I = [1 0; 0 1]     # local variable
    x1 .= [0.0, 0.0]   # assign
    v .= [1.0, 0.0]
    for (i, o) in enumerate(obs)
        x{i + 1} .= x{i} + v             # dynamic variable creation
        dv ~ MvNormal([0, 0], 0.1 * I)   # sample
        v .= v + dv
        o => MvNormal(x{i + 1}, 0.5 * I) # observe
    end
end
```

The `@smc` macro generates functions that propagate a DataFrame `particles` of weighted samples through SMC kernels. Particle variables are stored as columns (e.g., `particles.x`), with log-weights in `particles.weights`. Resampling occurs when the effective sample size drops below `ess_perc_min`. The macro generates two SMC functions:

```julia
# In-place (modifies existing particles)
function model!(args...; particles, kernels=nothing, proposals=nothing, 
                ess_perc_min=0.5, compute_evidence=true, show_progress=true)

# Sampling (creates new particles)
function model(args...; n_particles=1000, kernels=nothing, proposals=nothing,
               ess_perc_min=0.5, compute_evidence=true, show_progress=true)
```
Pass external data as function arguments. Both functions return the evidence (log-probability of observations) by default.

Within `@smc`, the following operators are available:

- **Particle assignment:** `x .= expr` broadcasts `expr` to `particles.x`.
- **Sampling:** `x ~ SMCKernel(args)` samples to `particles.x` and updates weights.
- **Observation:** `expr => SMCKernel(args)` updates weights based on observing `expr`.
- **MH Move:** `x << Proposal(args)` performs Metropolis-Hastings moves on `particles.x`.
Both `expr` and `args` can reference `particles.x` as `x`.

Regular Julia constructs are supported:

- **Assignment (`=`):** Creates local variables (not stored in `particles`).
- **For loops:** Supports `for i in collection` and tuple destructuring, where `collection` does not involve particle variables.
- **Conditionals:** `if condition` where `condition` does not involve particle variables.

Additional features:

- **Array indexing:** `x[i]` maps over `particles.x`.
- **Index interpolation:** `x{i}` creates dynamic variable names (e.g., `x1`, `x2`, ...).

## SMCKernel type

`SMCKernel` represents a (random weight) importance sampler:

- `sampler`: `(args...) -> sample` — generates samples
- `weighter`: `(args..., sample) -> log_weight` — computes log weights (can be `nothing` for uniform)
- `logpdf`: `(args..., sample) -> log_pdf` — evaluates log-density

Every `SMCKernel` represents a stochastic kernel given by averaging samples over weights:

```math
\int_{w} \text{weighter}(w \mid \text{args}, x) \text{sampler}(x \mid \text{args}) dw
```

The `logpdf` is the log-density of this kernel.

Default kernels are provided for major distributions from Distributions.jl (see `smc_kernels.jl`), accessible by name in `@smc`. Custom kernels can be defined using `SMCKernel` and passed as named tuples to SMC functions.

## MH moves

Resampling reduces particle diversity for early-sampled variables. MH moves can restore diversity. Example:

```julia
@smc function linear_regression(xs, ys)
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)
    for (x, y) in zip(xs, ys)
        y => Normal(α + β * x, 1.0)
        if resampled
            α << autoRW()
            β << autoRW()
        end
    end
end
```

**Note:** Moves require that particle variables are not overwritten before the move, as the MH acceptance ratio depends on previous values.

Moves are computationally expensive. Use them conditionally, based on the state of the particle approximation. The following variables are available within `@smc`:

- `particles`: The `particles` DataFrame
- `resampled`: Boolean, true if resampling occurred in the previous step
- `ess_perc`: Current effective sample size (percent)
- `evidence`: Current accumulated log-probability

MH kernels are functions `Proposal(particles, targets, args...)` returning a DataFrame of proposals and a vector of log proposal ratios.

Available proposals:

- `RW(step_size)`: Symmetric random walk
- `autoRW(min_step)`: Random walk with empirically calibrated covariance

Joint updates are supported:

```julia
(α, β) << autoRW()
```

## Utility functions

- `sample_particles(particles, n; replace=false)`: Draw `n` samples by weight
- `describe_particles(particles)`: Summarize all variables
- `exp_norm(weights)`: Normalize log-weights to probabilities
- `@E(f, particles)`: Compute weighted expectation of function `f` over particle variables

Examples:

```julia
@E(x -> x, particles)         # E[x]
@E(x -> x == 1, particles)    # P[x == 1]
@E((x, y) -> x + y, particles) # E[x + y]
```

## Performance tips

- Main bottleneck is resampling; many distinct variables (>10k) slow performance.
- Overwriting particle variables marginalizes them.
- Use static, concrete types (e.g., `StaticArrays.SVector`) for efficiency.
- Moves are expensive; use only when necessary.

## Saving particle history
Monitor SMC progress by saving snapshots:

```julia
ess_list = []
history = []

@smc function ssm(observations, ess_list, history)
    I = [1 0; 0 1]
    x .= [0.0, 0.0]
    v .= [1.0, 0.0]

    for obs in observations
        # Save snapshot
        push!(ess_list, ess_perc)
        push!(history, particles.x)

        x .= x + v
        dv ~ MvNormal([0,0], 0.1*I)
        v .= v + dv
        obs => MvNormal(x, 0.5*I)
    end
end
```
