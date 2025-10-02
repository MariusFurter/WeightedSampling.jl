# WeightedSampling.jl AI Assistant Instructions

WeightedSampling.jl is a macro-based Sequential Monte Carlo (SMC) library for Julia that transforms probabilistic programs into efficient particle filters.

## Core Architecture

### The `@smc` Macro System

- **Primary abstraction**: `@smc function model_name(args...)` transforms Julia functions into SMC samplers
- **Generates two functions**: `model(args...; n_particles=1000, ...)` (creates particles) and `model!(args...; particles, ...)` (in-place updates)
- **Particle storage**: DataFrames where columns are variables and `particles.weights` stores log-weights
- **Key operators**:
  - `x ~ Distribution(args)` - sampling with weight updates
  - `x => Distribution(args)` - observation/conditioning
  - `x .= expr` - broadcast assignment to all particles
  - `x << Proposal(args)` - Metropolis-Hastings moves

### Module Structure (`src/`)

- `WeightedSampling.jl` - main module exports
- `rewrites.jl` - macro expansion logic (symbol replacement, expression parsing)
- `smc_kernels.jl` - `SMCKernel` struct and default distribution kernels
- `resampling.jl` - ESS-based resampling algorithms
- `moves.jl` - MCMC move implementations
- `move_kernels.jl` - proposal distributions for MH moves
- `utils.jl` - particle utilities (`describe_particles`, `@E` macro, `sample_particles`)

## Development Patterns

### Writing `@smc` Models

```julia
@smc function model_name(data, hyperparams)
    # Local variables (not in particles)
    I = [1 0; 0 1]

    # Particle sampling
    θ ~ Normal(0, 1)

    # Sequential processing
    for obs in data
        # State updates
        x .= x + noise

        # Sampling with indexing
        noise ~ MvNormal(zeros(2), 0.1*I)

        # Conditioning/observation
        obs => MvNormal(x, 0.5*I)

        # Optional MCMC moves after resampling
        if resampled
            θ << autoRW()
        end
    end
end
```

### Symbol Resolution Rules

- Variables inside `@smc` become particle columns unless explicitly local
- Use `x .= expr` for particle assignment, `x = expr` for local variables
- Index interpolation: `x{i}` creates dynamic variable names (`x1`, `x2`, etc.)
- Array indexing: `x[i]` broadcasts over particle arrays

### Testing Conventions

- Use exact closed-form solutions when available (see `test/closed_form_conditioning.jl`)
- Validate both parameter estimates and evidence values
- Set `Random.seed!(42)` for reproducibility
- Use high particle counts (`100_000`) for accurate convergence tests

### Custom Kernels

Create `SMCKernel(sampler, weighter, logpdf)` where:

- `sampler: (args...) -> sample`
- `weighter: (args..., sample) -> log_weight` (optional, defaults to uniform)
- `logpdf: (args..., sample) -> log_density`

Pass custom kernels via `kernels=(custom_kernel=my_kernel,)` parameter.

## Key Dependencies

- `MacroTools.jl` for macro manipulation (`@capture`, `striplines`)
- `DataFrames.jl` for particle storage
- `Distributions.jl` for built-in distribution support
- `StatsBase.jl` for sampling utilities

## Common Workflows

### Running Models

```julia
# Basic usage
particles, evidence = model(data, n_particles=1000, ess_perc_min=0.5)

# Analysis
describe_particles(particles)  # Summary statistics
posterior_mean = @E(x -> x^2, particles)  # Weighted expectations
samples = sample_particles(particles, 1000)  # Resampling
```

### Adding New Distributions

Add to `default_kernels` in `smc_kernels.jl` using `@generate_kernels` macro with explicit parameter names.

### Debugging Macro Expansion

Use `@macroexpand @smc function ...` to inspect generated code. The macro transforms expressions in `rewrites.jl` using symbol replacement functions.
