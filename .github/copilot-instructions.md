# WeightedSampling.jl AI Assistant Instructions

WeightedSampling.jl is a macro-based Sequential Monte Carlo (SMC) library for Julia that transforms probabilistic programs into efficient particle filters.

## Core Architecture

### The `@model` Macro System

- **Primary abstraction**: `@model function model_name(args...)` compiles model code into a `ParticleTransformer` (typically a `Sequence`)
- **Execution pattern**: `model = model_name(args...)`; `state = SMCState(n_particles)`; `run!(model, state)`
- **Particle storage**: `state.store` (default backend `ColumnStore`) holds particle columns, and `state.weights` stores log-weights
- **Key operators**:
  - `x ~ Distribution(args)` - sampling with weight updates
  - `x => Distribution(args)` - observation/conditioning
  - `x .= expr` - broadcast assignment to all particles
  - `x << Proposal(args)` - Metropolis-Hastings moves

### Module Structure (`src/`)

- `WeightedSampling.jl` - main module exports
- `stores.jl` - particle storage backends (`AbstractParticleStore`, `ColumnStore`)
- `types.jl` - core runtime types (`SMCState`, `WeightedKernel`) and execution helpers
- `transformers.jl` - transformer implementations (`Assign`, `Sample`, `Observe`, `Move`, `Loop`, etc.)
- `rewrites.jl` - `@model` macro expansion and DSL rewriting
- `resampling.jl` - ESS-based resampling algorithms
- `move_kernels.jl` - proposal distributions for MH moves
- `default_kernels.jl` - default distribution kernels
- `utils.jl` - particle utilities (`describe`, `@E` macro, `sample`)

## Development Patterns

### Writing `@model` Models

```julia
@model function model_name(data, hyperparams)
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

- Variables inside `@model` become particle columns unless explicitly local
- Use `x .= expr` for particle assignment, `x = expr` for local variables
- Index interpolation: `x{i}` creates dynamic variable names (`x_1`, `x_2`, etc.)
- Array indexing: `x[i]` broadcasts over particle arrays

### Testing Conventions

- Use exact closed-form solutions when available (see tests under `test/`, especially `transformers_test.jl` and `macro_test.jl`)
- Validate both parameter estimates and evidence values
- Set `Random.seed!(42)` for reproducibility
- Use high particle counts (`100_000`) for accurate convergence tests

### Custom Kernels

Create `WeightedKernel(sampler, weighter, logpdf)` where:

- `sampler: (args...) -> sample`
- `weighter: (args..., sample) -> log_weight` (optional, defaults to uniform)
- `logpdf: (args..., sample) -> log_density`

Pass custom kernels via `kernels=(custom_kernel=my_kernel,)` parameter.

## Key Dependencies

- `MacroTools.jl` for macro manipulation (`@capture`, `striplines`)
- `DataFrames.jl` for result export/analysis utilities
- `Distributions.jl` for built-in distribution support
- `StatsBase.jl` for sampling utilities

## Common Workflows

### Running Models

```julia
# Basic usage
model = model_name(data)
state = SMCState(1000)
run!(model, state)

# Analysis
describe(state)  # Summary statistics
posterior_mean = @E(x -> x^2, state)  # Weighted expectations
samples = sample(state, 1000)  # Weighted sample
```

### Adding New Distributions

Add to `default_kernels` in `default_kernels.jl` using `@generate_kernels` macro with explicit parameter names.

### Debugging Macro Expansion

Use `@macroexpand @model function ...` to inspect generated code. The macro transforms expressions in `rewrites.jl` using symbol replacement functions.
