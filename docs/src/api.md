# API Reference

```@meta
CurrentModule = WeightedSampling
```

## Core Types

```@docs
SMCState
WeightedKernel
```

## Execution

```@docs
run!
```

## The `@model` macro

```@docs
@model
```

## Utility

```@docs
@E
expectation
exp_norm
describe
sample
```

## MH proposal kernels

```@docs
default_proposals
autoRW
RW
```

## Default Distribution Kernels

```@docs
default_kernels
```

## Distribution kernels

WeightedSampling.jl provides kernels for many Distributions.jl distributions.
These are available through `default_kernels` and are resolved by name in
`@model` bodies.

### Continuous distributions
`Normal(μ, σ)`, `Beta(α, β)`, `Gamma(α, θ)`, `Exponential(θ)`, `Uniform(a, b)`, `Cauchy(μ, σ)`, `Laplace(α, θ)`, `LogNormal(μ, σ)`, `Weibull(α, θ)`, `Chi(ν)`, `Chisq(ν)`, `TDist(ν)`, `FDist(ν1, ν2)`, `Pareto(α, θ)`, `Rayleigh(σ)`, `Gumbel(μ, θ)`, `Frechet(α, θ)`, `InverseGamma(α, θ)`, `LogitNormal(μ, σ)`, `Logistic(μ, θ)`, `SkewNormal(ξ, ω, α)`, `SkewedExponentialPower(μ, σ, p, α)`, `VonMises(μ, κ)`, `GeneralizedPareto(μ, σ, ξ)`, `NoncentralChisq(ν, λ)`, `NoncentralF(ν1, ν2, λ)`, `NoncentralT(ν, λ)`, `NormalCanon(η, λ)`

### Discrete distributions  
`Bernoulli(p)`, `BernoulliLogit(logitp)`, `Binomial(n, p)`, `BetaBinomial(n, α, β)`, `Categorical(p)`, `DiscreteUniform(a, b)`, `Geometric(p)`, `Hypergeometric(s, f, n)`, `NegativeBinomial(r, p)`, `Poisson(λ)`, `PoissonBinomial(p)`, `DiscreteNonParametric(xs, ps)`, `Dirac(x)`

### Multivariate distributions
`MvNormal(μ, Σ)`, `MvNormalCanon(h, J)`, `MvLogNormal(μ, Σ)`, `MvLogitNormal(μ, Σ)`, `Multinomial(n, p)`, `Dirichlet(α)`, `LKJ(d, η)`, `LKJCholesky(d, η)`, `Wishart(ν, S)`, `InverseWishart(ν, Ψ)`

### Matrix-valued distributions
`MatrixNormal(M, U, V)`, `MatrixBeta(p, n1, n2)`, `MatrixFDist(n1, n2, B)`, `MatrixTDist(ν, M, Σ, Ω)`

## Advanced usage

### Custom kernels

You can define custom kernels for distributions not included in the package:

```julia
using Distributions

# Custom truncated normal kernel
TruncatedNormal = WeightedKernel(
    (μ, σ, a, b) -> rand(Truncated(Normal(μ, σ), a, b)),
    nothing,  # uniform weights
    (μ, σ, a, b, x) -> logpdf(Truncated(Normal(μ, σ), a, b), x)
)

@model function model()
    θ ~ TruncatedNormal(0.0, 1.0, -2.0, 2.0)
end

# Build and run
transformer = model(kernels=(TruncatedNormal=TruncatedNormal,))
state = SMCState(500)
run!(transformer, state)
```

### Custom proposals

You can define custom MH proposals using the following signature:

```julia
function Proposal(state, targets, params...)
    # Return (proposed_changes, log_ratios)
end
```

Here `proposed_changes` is typically a `Dict{Symbol,<:AbstractVector}` and
`log_ratios` is a vector with entries
``\log q(x_\text{old} \mid x_\text{new}) - \log q(x_\text{new} \mid x_\text{old})``.