using Test
using Random
using Statistics
using Distributions

@testset "importance_kernel" begin
    Random.seed!(42)

    proposal = Normal(0.0, 2.0)
    target = Normal(1.0, 1.0)
    kernel = importance_kernel(proposal, target)

    N = 200_000
    xs = [kernel.sampler() for _ in 1:N]
    logws = kernel.weighter.(xs)

    # logpdf field reports the target's density, not the proposal's.
    @test kernel.logpdf.(xs) == logpdf.(target, xs)
    @test logws == logpdf.(target, xs) .- logpdf.(proposal, xs)

    w = exp_norm(logws)
    est_mean = sum(xs .* w)
    @test isapprox(est_mean, mean(target), atol=0.05)

    # Both proposal and target are normalized densities, so the estimated
    # evidence (normalizing constant) should be close to 1, i.e. log ≈ 0.
    log_evidence = logsumexp(logws) - log(N)
    @test isapprox(log_evidence, 0.0, atol=0.05)
end
