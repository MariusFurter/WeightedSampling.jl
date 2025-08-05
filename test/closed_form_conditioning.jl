using DrawingInferences
using Distributions
using CairoMakie

### Beta-binomial model

function makeBeta11Sampler()
    return WeightedSampler(
        (rng) -> rand(rng, Beta(1, 1)),
        (p::Float64) -> logpdf(Beta(1, 1), p),
        (p::Float64) -> 0.0,
        () -> Float64
    )
end

Beta11Sampler = makeBeta11Sampler()

function makeBinomial7Observer()
    return WeightedSampler(
        (p::Float64, rng) -> 0.0,
        (x::Int64) -> (p::Float64) -> logpdf(Binomial(10, p), x),
        (p::Float64) -> logpdf(Binomial(10, p), 7),
        () -> Float64
    )
end

Binomial7Observer = makeBinomial7Observer()



#Issue: (Partial) literal parameters to samplers

@fk function beta_binomial()
    p = BetaSampler(1.0, 1.0)
    y = BinomialSampler(10, p)
end

#Issue: Literals on RHS of assignments

@fk function beta_binomial()
    alpha = 1.0
    p = BetaSampler(alpha, 1.0)
    y = BinomialSampler(10, p)
end

#Issue: Observers can't weight by inputs.
# --> Let weighter be a function from (inputs, output) to weight
# --> problem arises when input is overwritten.

@fk function beta_binomial()
    p = Beta11Sampler()
    y = Binomial7Observer(p)
end


#Issue: Nothing / Empty tuple not supported as output type
# --> Overwrite base method, or implement own default value function


fk = beta_binomial()

model = SMCModel(fk)

N_particles = 2^15
nthreads = 1# Threads.nthreads()
fullOutput = false
essThreshold = 2.0

smcio = SMCIO{model.particle,model.pScratch}(N_particles, length(fk), nthreads, fullOutput, essThreshold)

@time smc!(model, smcio)

smcio.zetas

ps = [p.p.p for p in smcio.zetas]
ws = smcio.ws

hist(ps, weights=smcio.ws, normalization=:pdf, bins=100)

xs = 0:0.01:1.0
lines!(xs, pdf.(p_given_y(10, 7, 1, 1), xs))
current_figure()

#Closed form solution
function p_given_y(n, k, a, b)
    # n: number of trials
    # k: number of successes
    # a: alpha parameter of the Beta prior
    # b: beta parameter of the Beta prior
    # Returns the posterior probability of p given y
    return Beta(k + a, n - k + b)
end