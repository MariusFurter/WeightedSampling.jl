# API Reference

```@meta
CurrentModule = WeightedSampling
```

## Core Types

```@docs
SMCKernel
```

## The `@smc` macro

```@docs
@smc
```

## Utility

```@docs
@E
exp_norm
describe_particles
sample_particles
diversity
```

## MH proposal kernels

```@docs
autoRW
RW
```

## Distribution kernels

WeightedSampling.jl provides kernels for all major distributions from Distributions.jl. These can be used directly by name in `@smc` functions:

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

You can define custom SMC kernels for distributions not included in the package:

```julia
using Distributions

# Custom truncated normal kernel
TruncatedNormal = SMCKernel(
    (μ, σ, a, b) -> rand(Truncated(Normal(μ, σ), a, b)),
    nothing,  # uniform weights
    (μ, σ, a, b, x) -> logpdf(Truncated(Normal(μ, σ), a, b), x)
)

@smc function model()
    θ ~ TruncatedNormal(0.0, 1.0, -2.0, 2.0)
end

# Pass custom kernel to the model
particles, evidence = model(
    kernels=(TruncatedNormal=TruncatedNormal,)
)
```

### Custom proposals

You can defined custom MH proposals using the following signature:

```julia
function Proposal(particles, targets, params...)
    # Return (proposed_changes::DataFrame, log_ratios::Vector)
end
```

Here `log_ratios` is a vector with entries ``\log q(x_\text{old} \mid x_\text{new}) - \log q(x_\text{new} \mid x_\text{old})``, where ``q`` is the density of the proposal kernel.