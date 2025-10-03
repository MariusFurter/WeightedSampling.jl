# WeightedSampling.jl

[![Build Status](https://github.com/MariusFurter/WeightedSampling.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MariusFurter/WeightedSampling.jl/actions/workflows/CI.yml?query=branch%3Amain)

**WeightedSampling.jl** provides a macro-based interface for Sequential Monte Carlo (SMC) in Julia. It enables concise, readable probabilistic programs that are transformed into efficient particle filters with resampling and weight management.

## Features

- Intuitive `@smc` macro for SMC model specification
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

Here's a simple linear regression example:

```julia
using WeightedSampling

@smc function linear_regression(xs, ys)
    α ~ Normal(0, 1)  # sample prior
    β ~ Normal(0, 1)
    
    for (x, y) in zip(xs, ys)
        y => Normal(α + β * x, 1.0)  # observe data
    end
end

# Generate some data
xs = 1:10
ys = 2.0 .+ 0.5 .* xs .+ randn(length(xs))

# Run SMC inference
particles, evidence = linear_regression(xs, ys, n_particles=1000)

# Analyze results
describe_particles(particles)
```

## The `@smc` Macro

The core of **WeightedSampling.jl** is the `@smc` macro, which transforms a Julia function into a Sequential Monte Carlo (SMC) model with automatic weight management, resampling, and support for Metropolis-Hastings (MH) moves.

### Key Operators

- **Particle assignment:** `x .= expr` broadcasts `expr` to all particles
- **Sampling:** `x ~ SMCKernel(args)` samples from a distribution or kernel and updates particle weights
- **Observation:** `expr => SMCKernel(args)` conditions on observed data, updating weights accordingly
- **MH Move:** `x << Proposal(args)` applies MH moves to particles

### Generated Functions

SMC models defined with `@smc` are compiled to two main functions:
- `model!(args...; particles, ...)` *(in-place update of existing particles)*
- `model(args...; n_particles=..., ...)` *(creates and returns new particles)*

## Navigation

- [Usage Guide](usage_guide.md): Detailed usage instructions and examples
- [API Reference](api.md): Complete API documentation
- [Examples](examples.md): Additional examples and use cases
