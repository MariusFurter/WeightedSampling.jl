using WeightedSampling
using SequentialMonteCarlo
using RNGPool
using BenchmarkTools
using Random
using Printf

include(joinpath(@__DIR__, "..", "simulate.jl"))

# Same model/params as the other benchmarks/ssm/ scripts:
#     x(0) ~ Normal(0, x0_std)
#     x(t) = a * x(t-1) + w(t),   w(t) ~ Normal(0, q)
#     y(t) ~ Normal(x(t), r)
const A, Q, R, X0_STD = 0.9, 1.0, 0.5, 1.0

# ---------------------------------------------------------------------------
# WeightedSampling
# ---------------------------------------------------------------------------

"""
Prefix model (used only to populate a realistic `:x` column before timing a
single isolated update); identical to `WeightedSampling/lgssm1d.jl`'s model.
"""
@model function lgssm1d_prefix(data, a, q, r, x0_std)
    x ~ Normal(0.0, x0_std)
    for y in data
        x ~ Normal(a * x, q)
        y => Normal(x, r)
    end
end

"""
Single mutate+observe(+resample) step, operating directly on a pre-existing
`:x` column created by an earlier (unrelated) model run on the same state
(see repo memory notes on this pattern). `~`/`=>` each auto-insert a
`Resample()` (forced to always trigger via `ess_perc_min=1.0`), so timing
`apply!`/`run!` of this model reproduces the exact per-iteration cost of the
main loop's body (transition + observation + 2x forced resample).
"""
@model function lgssm1d_ws_step(y, a, q, r)
    x ~ Normal(a * x, q)
    y => Normal(x, r)
end

"""
    bench_ws(N; prefix_T=50, seed=42)

Benchmark ONE mutate+observe+resample update of the WeightedSampling model at
`N` particles, forcing resampling every step (`ess_perc_min=1.0`).
"""
function bench_ws(N::Int; a=A, q=Q, r=R, x0_std=X0_STD, seed=42, prefix_T=50)
    Random.seed!(seed)
    _, data = simulate_lgssm1d(prefix_T + 1, a, q, r, x0_std)
    prefix_data = data[1:prefix_T]
    y = data[prefix_T+1]

    step_model = lgssm1d_ws_step(y, a, q, r)

    # Warm up JIT (small state) before building the real N-particle state.
    warmup_state = SMCState(100; ess_perc_min=1.0)
    run!(lgssm1d_prefix(prefix_data[1:2], a, q, r, x0_std), warmup_state)
    run!(step_model, warmup_state)

    # Build the real state and populate :x via the (untimed) prefix run.
    state = SMCState(N; ess_perc_min=1.0)
    run!(lgssm1d_prefix(prefix_data, a, q, r, x0_std), state)

    return @benchmark run!($step_model, $state)
end

# ---------------------------------------------------------------------------
# SequentialMonteCarlo.jl
# ---------------------------------------------------------------------------

mutable struct Float64Particle
    x::Float64
    Float64Particle() = new()
end

@inline function Base.:(==)(x::Float64Particle, y::Float64Particle)
    return x.x == y.x
end

"""
Same `SMCModel` construction as `SequentialMonteCarlo/lGModel.jl` (not
`include`-d directly here since that script does its own `Pkg.activate`).
"""
function make_lgssm1d_model(ys::Vector{Float64}, a, q, r, x0_std)
    n::Int64 = length(ys)
    logncG::Float64 = -0.5 * log(2 * π * r^2)
    invRover2::Float64 = 0.5 / r^2

    @inline function lG(p::Int64, particle::Float64Particle, ::Nothing)
        @inbounds v::Float64 = particle.x - ys[p]
        return logncG - v * invRover2 * v
    end

    @inline function M!(newParticle::Float64Particle, rng::RNG, p::Int64,
        particle::Float64Particle, ::Nothing)
        if p == 1
            newParticle.x = x0_std * randn(rng)
        else
            newParticle.x = a * particle.x + q * randn(rng)
        end
    end

    return SMCModel(M!, lG, n, Float64Particle, Nothing)
end

"""
    _smc_step!(smcio, model, p, engine)

Perform ONE iteration `p` of `SequentialMonteCarlo.jl`'s internal
`_smcSerial!` loop body (mutate, log-weight/normalize, ESS, resample) directly
on an already-populated `smcio`, using that package's real (but unexported)
per-step routines. This is a manual, line-for-line extraction of one pass of
the `for p = 1:smcio.n` loop in `serial.jl`'s `_smcSerial!` (not a
reimplementation of the algorithm) -- there's no public API for this, since
the per-step internals are all unexported.
"""
function _smc_step!(smcio::SMCIO{Particle}, model::SMCModel, p::Int64, engine) where Particle
    zetas = smcio.zetas
    zetaAncs = smcio.internal.zetaAncs
    as = smcio.internal.as
    lws = smcio.internal.lws
    ws = smcio.ws
    pScratch = smcio.internal.particleScratch

    if p > 1
        SequentialMonteCarlo._copyParticles!(zetaAncs, zetas, as)
        smcio.internal.oldEves .= smcio.eves
        SequentialMonteCarlo._setEves!(smcio.eves, smcio.internal.oldEves, as)
    end

    SequentialMonteCarlo._mutateParticles!(zetas, engine, p, model.M!, zetaAncs, pScratch)
    SequentialMonteCarlo._logWeightParticles!(lws, p, model.lG, zetas, pScratch,
        p == 1 || smcio.resample[p-1], ws)

    maxlw = maximum(lws)
    ws .= exp.(lws .- maxlw)
    mws = sum(ws) / smcio.N
    smcio.internal.maxlw = maxlw
    smcio.internal.mws = mws
    smcio.logZhats[p] = (p == 1 ? 0.0 : smcio.logZhats[p-1]) + maxlw + log(mws)
    ws ./= mws
    smcio.esses[p] = smcio.N / mapreduce(x -> x * x, +, ws)
    smcio.Vhat1s[p] = SequentialMonteCarlo._Vhat1Serial(smcio)

    p < smcio.n && SequentialMonteCarlo._resampleSerial!(smcio, p, false)
    return nothing
end

"""
    bench_smc(N; prefix_T=50, seed=42)

Benchmark ONE mutate+weight+resample update of the SequentialMonteCarlo.jl
model at `N` particles, forcing resampling every step (`essThreshold=1.0`).
Mirrors `bench_ws`'s exact methodology: build a realistic `smcio` state via an
(untimed) prefix "run" of `prefix_T` steps (via `_smc_step!`, since `smc!`
itself can't be stopped partway through and resumed), then time ONE more step
in place on that same `smcio` via `@benchmark` (repeated calls each advance
the filter by one step, same in-place-mutation pattern as `bench_ws`'s
repeated `run!` calls). `n = prefix_T + 2` (not `+1`) so the timed step index
`prefix_T+1` still satisfies `p < smcio.n` and therefore triggers a forced
resample, matching `bench_ws`'s single step which also always resamples.
"""
function bench_smc(N::Int; a=A, q=Q, r=R, x0_std=X0_STD, seed=42, prefix_T=50)
    if Threads.nthreads() != 1
        error("This benchmark requires Threads.nthreads()==1 (RNGPool requirement); " *
              "restart Julia with `-t 1`.")
    end

    setRNGs(0)
    Random.seed!(seed)
    n = prefix_T + 2
    _, data = simulate_lgssm1d(n, a, q, r, x0_std)
    essThreshold = 1.0
    model = make_lgssm1d_model(data, a, q, r, x0_std)
    engine = getRNG()

    # Warm up JIT (small state) before building the real N-particle smcio.
    warmup_smcio = SMCIO{Float64Particle,Nothing}(100, n, 1, false, essThreshold)
    for p in 1:(prefix_T+1)
        _smc_step!(warmup_smcio, model, p, engine)
    end

    # Build the real state and populate it via the (untimed) prefix run.
    smcio = SMCIO{Float64Particle,Nothing}(N, n, 1, false, essThreshold)
    for p in 1:prefix_T
        _smc_step!(smcio, model, p, engine)
    end

    p = prefix_T + 1
    return @benchmark _smc_step!($smcio, $model, $p, $engine)
end

# ---------------------------------------------------------------------------
# Comparison sweep
# ---------------------------------------------------------------------------

function run_comparison(; N_values=(1_000, 10_000, 100_000))
    if Threads.nthreads() != 1
        @warn "Running with Threads.nthreads()=$(Threads.nthreads()); " *
              "launch Julia with `-t 1` for the SequentialMonteCarlo.jl half " *
              "of this comparison to work at all (RNGPool requirement)."
    end

    @printf("%-12s %-16s %-14s %-16s %-14s\n", "N", "WS median (us)", "WS alloc (KiB)",
        "SMC median (us)", "SMC alloc (KiB)")
    for N in N_values
        ws_bench = bench_ws(N)
        ws_median_us = median(ws_bench).time / 1e3
        ws_alloc_kib = ws_bench.memory / 2^10

        smc_bench = bench_smc(N)
        smc_median_us = median(smc_bench).time / 1e3
        smc_alloc_kib = smc_bench.memory / 2^10

        @printf("%-12d %-16.3f %-14.2f %-16.3f %-14.2f\n", N, ws_median_us, ws_alloc_kib,
            smc_median_us, smc_alloc_kib)
        @printf("RESULT,bench_single_update,N=%d,ws_median_us=%.4f,ws_alloc_kib=%.4f,smc_median_us=%.4f,smc_alloc_kib=%.4f\n",
            N, ws_median_us, ws_alloc_kib, smc_median_us, smc_alloc_kib)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_comparison()
end
