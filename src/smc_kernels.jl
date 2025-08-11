struct SMCKernel{S,L,W}
    sampler::S
    logpdf::L
    weighter::W
end

macro from_distribution(dist_type)
    fields = fieldnames(eval(dist_type))
    quote
        SMCKernel(
            ($(fields...),) -> rand($(dist_type)($(fields...))),
            ($(fields...), x) -> logpdf($(dist_type)($(fields...)), x),
            nothing
        )
    end
end

macro generate_kernels(distributions...)
    tuple_entries = []

    for dist in distributions
        push!(tuple_entries, :($dist = @from_distribution($dist)))
    end

    quote
        ($(tuple_entries...),)
    end
end

default_kernels = @generate_kernels(
    Beta, BernoulliLogit, Bernoulli, BetaBinomial, Binomial,
    Categorical, Cauchy, Chi, Chisq,
    Dirac, Dirichlet, DiscreteNonParametric, DiscreteUniform,
    Exponential,
    FDist, Frechet,
    Gamma, GeneralizedPareto, Geometric, Gumbel,
    Hypergeometric,
    InverseGamma, InverseWishart,
    LKJ, LKJCholesky, Laplace, LogNormal, Logistic, LogitNormal,
    MatrixBeta, MatrixFDist, MatrixNormal, MatrixTDist, MvLogNormal, MvLogitNormal, MvNormal, MvNormalCanon,
    NegativeBinomial, NoncentralChisq, NoncentralF, NoncentralT, Normal, NormalCanon,
    Pareto, Poisson, PoissonBinomial,
    Rayleigh,
    SkewNormal, SkewedExponentialPower,
    TDist,
    Uniform,
    VonMises,
    Weibull, Wishart,
)
