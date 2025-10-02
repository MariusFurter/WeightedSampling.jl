"""
    SMCKernel{S,L,W}

Represents a (random weight) importance sampler.

# Fields
- `sampler::S`: Function `(args...) -> sample` that generates samples
- `weighter::W`: Function `(args..., sample) -> log_weight` that computes log weights. Can be random. Use `nothing` for uniform weights.
- `logpdf::L`: Function `(args..., sample) -> log_density` that evaluates log-density

Every `SMCKernel` represents a stochastic kernel given by averaging samples over weights:
```math
\\int_{w} \\text{weighter}(w \\mid \\text{args}, x) \\text{sampler}(x \\mid \\text{args}) dw
```
The `logpdf` function is the density of this kernel.

# Constructor
```julia
SMCKernel(sampler, weighter, logpdf)
```

# Examples
```julia
using Distributions

# Custom truncated normal kernel
TruncatedNormal = SMCKernel(
    (μ, σ, a, b) -> rand(Truncated(Normal(μ, σ), a, b)),
    nothing,  # uniform weights
    (μ, σ, a, b, x) -> logpdf(Truncated(Normal(μ, σ), a, b), x)
)

# Use in @smc model
@smc function model(data)
    θ ~ TruncatedNormal(0.0, 1.0, -2.0, 2.0)
    for y in data
        y => Normal(θ, 0.5)
    end
end
```

See also: [`@smc`](@ref)
"""
struct SMCKernel{S,L,W}
    sampler::S
    weighter::W
    logpdf::L
end

macro from_distribution(dist_type, args...)
    fields = [args...]
    unique_fields = [gensym(string(field)) for field in fields]

    quote
        SMCKernel(
            ($(unique_fields...),) -> rand($(dist_type)($(unique_fields...))),
            nothing,
            ($(unique_fields...), x) -> logpdf($(dist_type)($(unique_fields...)), x),
        )
    end
end

macro generate_kernels(distributions...)
    tuple_entries = []

    for dist in distributions
        if isa(dist, Expr) && dist.head == :call
            dist_type = dist.args[1]
            args = dist.args[2:end]
            push!(tuple_entries, :($dist_type = @from_distribution($dist_type, $(args...))))
        else
            error("Distributions must be specified with explicit argument names, e.g., Beta(α, β) or Dirichlet(alpha).")
        end
    end

    quote
        ($(tuple_entries...),)
    end
end

# Default kernels from Distributions.jl
default_kernels = @generate_kernels(
    Beta(α, β), BernoulliLogit(logitp), Bernoulli(p), BetaBinomial(n, α, β), Binomial(n, p),
    Categorical(p), Cauchy(μ, σ), Chi(ν), Chisq(ν),
    Dirac(x), Dirichlet(α), DiscreteNonParametric(xs, ps), DiscreteUniform(a, b),
    Exponential(θ),
    FDist(ν1, ν2), Frechet(α, θ),
    Gamma(α, θ), GeneralizedPareto(μ, σ, ξ), Geometric(p), Gumbel(μ, θ),
    Hypergeometric(s, f, n),
    InverseGamma(α, θ), InverseWishart(ν, Ψ),
    LKJ(d, η), LKJCholesky(d, η), Laplace(α, θ), LogNormal(μ, σ), Logistic(μ, θ), LogitNormal(μ, σ),
    MatrixBeta(p, n1, n2), MatrixFDist(n1, n2, B), MatrixNormal(M, U, V), MatrixTDist(ν, M, Σ, Ω), MvLogNormal(μ, Σ), MvLogitNormal(μ, Σ), MvNormal(μ, Σ), MvNormalCanon(h, J), Multinomial(n, p),
    NegativeBinomial(r, p), NoncentralChisq(ν, λ), NoncentralF(ν1, ν2, λ), NoncentralT(ν, λ), Normal(μ, σ), NormalCanon(η, λ),
    Pareto(α, θ), Poisson(λ), PoissonBinomial(p),
    Rayleigh(σ),
    SkewNormal(ξ, ω, α), SkewedExponentialPower(μ, σ, p, α),
    TDist(ν),
    Uniform(a, b),
    VonMises(μ, κ),
    Weibull(α, θ), Wishart(ν, S)
)
