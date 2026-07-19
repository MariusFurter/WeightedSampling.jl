"""
Default `WeightedKernel`s wrapping `Distributions.jl` distributions: sampling
via `rand`, uniform weighting, density via `logpdf`.
"""

"""
    @from_distribution(dist_type, args...)

Build a `WeightedKernel` wrapping `Distributions.dist_type(args...)`: samples
via `rand`, uniform weighting (`weighter=nothing`), density via `logpdf`.
"""
macro from_distribution(dist_type, args...)
    fields = [args...]
    unique_fields = [gensym(string(field)) for field in fields]

    quote
        WeightedKernel(
            ($(unique_fields...),) -> rand($(dist_type)($(unique_fields...))),
            nothing,
            ($(unique_fields...), x) -> logpdf($(dist_type)($(unique_fields...)), x),
        )
    end
end

"""
    @generate_kernels(dist(args...), ...)

Build a `NamedTuple` mapping each distribution type name to a
`@from_distribution`-built `WeightedKernel`. Distributions must be given with
explicit argument names, e.g. `Beta(α, β)`.
"""
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

"""
    default_kernels

`NamedTuple` of built-in `WeightedKernel`s covering most `Distributions.jl`
distributions, keyed by distribution type name (e.g. `default_kernels.Normal`).
`@model`-generated functions merge a user-supplied `kernels` `NamedTuple` into
this fallback table (user entries override same-named defaults).
"""
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
