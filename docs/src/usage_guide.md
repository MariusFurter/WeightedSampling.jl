# Usage Guide

WeightedSampling.jl uses a macro DSL to build Sequential Monte Carlo programs.
The core flow is:

1. Define a model with `@model`.
2. Build a transformer by calling the model function.
3. Run it on an `SMCState`.

## Installation

The package is not registered yet. Install from GitHub:

```julia
using Pkg
Pkg.add("https://github.com/MariusFurter/WeightedSampling.jl")
```

## Quick Start

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

xs = 1:10
ys = 1.0 .- 0.5 .* xs .+ 0.5 .* randn(length(xs))

model = linear_regression(xs, ys)
state = SMCState(1_000)
run!(model, state)

describe(state)
posterior = sample(state, 200)
alpha_mean = @E(α -> α, state)
```

## Model DSL

Inside `@model`, these operators are available:

- `x .= expr`: broadcast assignment to particle variable `x`
- `x ~ KernelOrDistribution(args...)`: sample/update step
- `obs => KernelOrDistribution(args...)`: observation/conditioning step
- `x << Proposal(args...)`: MH move step

### Dynamic Variables

Use brace interpolation for dynamic names:

```julia
@model function ssm(obs)
    I2 = [1.0 0.0; 0.0 1.0]
    x{1} .= [0.0, 0.0]
    v .= [1.0, 0.0]
    for (i, o) in enumerate(obs)
        x{i + 1} .= x{i} + v
        dv ~ MvNormal([0.0, 0.0], 0.1 * I2)
        v .= v + dv
        o => MvNormal(x{i + 1}, 0.5 * I2)
    end
end
```

## Running A Model

`@model` generates a function that returns a `ParticleTransformer` (usually a
`Sequence`).

```julia
model = my_model(args...)
state = SMCState(1000)
run!(model, state)
```

`run!` works for models with and without move steps.

## State And Analysis

An `SMCState` stores particle columns and log-weights.

Common utilities:

- `describe(state)` for weighted summary statistics
- `sample(state, n)` for weighted posterior samples
- `@E(f, state)` for weighted expectations
- `DataFrame(state)` for raw particle export plus `:log_weight`

Examples:

```julia
@E(x -> x, state)
@E((x, y) -> x + y, state)
```

## Custom Kernels And Proposals

You can override defaults per model call:

```julia
custom_normal = WeightedKernel(
    (μ, σ) -> rand(Normal(μ, σ)),
    nothing,
    (μ, σ, x) -> logpdf(Normal(μ, σ), x),
)

model = linear_regression(xs, ys; kernels=(Normal=custom_normal,))
state = SMCState(1_000)
run!(model, state)
```

Proposals are provided through the `proposals` keyword similarly.
