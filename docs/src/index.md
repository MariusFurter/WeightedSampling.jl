# WeightedSampling.jl

[![Build Status](https://github.com/MariusFurter/WeightedSampling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MariusFurter/WeightedSampling.jl/actions/workflows/CI.yml?query=branch%3Amain)

**WeightedSampling.jl** provides a macro-based interface for Sequential Monte Carlo (SMC) in Julia. The `@model` macro compiles a probabilistic program into a particle-transformer program that you run on an `SMCState`.

## Features

- Intuitive `@model` macro for SMC model specification
- Automatic weight management and resampling
- Support for Metropolis-Hastings moves within SMC
- Flexible kernel and proposal definitions
- Utility functions for particle analysis and sampling

**Note:** This package is in early development. Use with caution. Feedback is welcome!

## Installation

This package is not yet registered. Install directly from GitHub:

```julia
using Pkg
Pkg.add("https://github.com/MariusFurter/WeightedSampling.jl")
```

## Quick Start

Here is a minimal runnable linear regression example:

```julia
using WeightedSampling
using Random

Random.seed!(42)

@model function linear_regression(xs, ys)
    α ~ Normal(0.0, 10.0)
    β ~ Normal(0.0, 10.0)
    for (x, y) in zip(xs, ys)
        y => Normal(α + β * x, 1.0)
        if resampled
            α << autoRW()
            β << autoRW()
        end
    end
end

# Generate synthetic data
xs = 1:10
ys = 1.0 .- 0.5 .* xs .+ 0.5 .* randn(length(xs))

# Build model and run SMC
model = linear_regression(xs, ys)
state = SMCState(1_000)
run!(model, state)

# Analyze results
describe(state)
posterior_mean_alpha = @E(α -> α, state)
```

## The `@model` Macro

The core of **WeightedSampling.jl** is the `@model` macro, which transforms a Julia function into a Sequential Monte Carlo (SMC) model with automatic weight management, resampling, and support for Metropolis-Hastings (MH) moves.

### Key Operators

- **Particle assignment:** `x .= expr` broadcasts `expr` to all particles.
- **Sampling:** `x ~ Distribution(args)` (or custom kernel name) samples and may update weights.
- **Observation:** `expr => Distribution(args)` conditions on data by updating weights.
- **MH move:** `x << Proposal(args)` applies a Metropolis-Hastings move.

### Generated Functions

`@model` compiles to a constructor function that returns a `ParticleTransformer`
(typically a `Sequence`). The usual workflow is:

1. Build the transformer: `model = my_model(args...)`
2. Create state: `state = SMCState(n_particles)`
3. Run it: `run!(model, state)`

`run!` is the recommended entry point for all models.

## Navigation

- [Usage Guide](usage_guide.md): Detailed usage instructions and examples
- [API Reference](api.md): Complete API documentation
- [Examples](examples.md): Additional examples and use cases
