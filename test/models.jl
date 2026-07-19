using DataFrames
using Distributions
using LinearAlgebra: I
using Random
using StatsBase: ProbabilityWeights, cov
using WeightedSampling

"""
    NormalKernel

`WeightedKernel` wrapping `Distributions.Normal`, mirroring the entries in
`WeightedSampling.default_kernels`. Uniform weighting (`weighter=nothing`),
since we haven't ported `Observe`/`Weight` transformers yet.
"""
const NormalKernel = WeightedKernel(
    (μ, σ) -> rand(Normal(μ, σ)),
    nothing,
    (μ, σ, x) -> logpdf(Normal(μ, σ), x),
)

"""
    rw_init_argfn(state)

Args for the initial sample `x_k(0) ~ Normal(0, 1)`. Scalars wrapped in `Ref`
rather than `fill`, so no length-N allocation is needed for arguments that
don't vary across particles.
"""
rw_init_argfn(state) = (Ref(0.0), Ref(1.0))

"""
    rw_step_argfn(colsym)

Returns a closure computing args for `x_k(t) ~ Normal(x_k(t-1), 1)`. Built via
a factory function (rather than inlined per call site) so that every step
sampling the same variable shares one concrete closure type — only the
captured `colsym` field differs. This keeps `Sequence`'s `Tuple` of steps from
growing type-diverse just because the walk runs for many steps; see the note
in the benchmark script about how this differs from what per-line macro
expansion will eventually generate.
"""
rw_step_argfn(colsym) = state -> (getcol(state.store, colsym), Ref(1.0))

"""
    build_random_walk(K::Int, T::Int)

Build a `Sequence` transformer simulating `K` independent 1D random walks,
each run for `T` steps:

    x_k(0) ~ Normal(0, 1)                        for k in 1:K
    x_k(t) ~ Normal(x_k(t-1), 1)     for t in 1:T, k in 1:K

Pure forward simulation, no observations/conditioning (Observe/Weight
transformers aren't implemented yet), so all particle weights stay at 0.
"""
function build_random_walk(K::Int, T::Int)
    steps = ParticleTransformer[]

    for k in 1:K
        push!(steps, Sample(Symbol(:x, k), NormalKernel, rw_init_argfn))
    end

    for t in 1:T, k in 1:K
        colsym = Symbol(:x, k)
        push!(steps, Sample(colsym, NormalKernel, rw_step_argfn(colsym)))
    end

    return Sequence(Tuple(steps))
end

"""
    rw_step_body(K::Int)

Builds the per-iteration body for `build_random_walk_loop`: a `Sequence` of
`K` update `Sample` steps (one per random-walk variable), independent of the
loop index `t`. Used as the `bodyfn` of a `Loop`, so it is (re)built once per
`t` — but always with the same concrete type — rather than the whole `T`-long
step list being unrolled into one flat `Tuple` up front.
"""
rw_step_body(K::Int) = t -> Sequence(Tuple(Sample(Symbol(:x, k), NormalKernel, rw_step_argfn(Symbol(:x, k))) for k in 1:K))

"""
    build_random_walk_loop(K::Int, T::Int)

Same model as `build_random_walk`, but the `T`-step update loop is expressed
as a `Loop` transformer (collection `1:T`) instead of being unrolled into `T*K`
explicit `Sample` steps at construction time. `K` is still unrolled, so this
isolates the effect of *not* unrolling the `T` loop specifically.
"""
function build_random_walk_loop(K::Int, T::Int)
    init_steps = Tuple(Sample(Symbol(:x, k), NormalKernel, rw_init_argfn) for k in 1:K)
    loop = Loop(state -> 1:T, rw_step_body(K))
    return Sequence((init_steps..., loop))
end

"""
    init_particles(N::Int)

Fresh particle store with `N` particles (default backend: `ColumnStore`).
Weights are held separately on `SMCState`, so the store starts with no
columns. Use `ColumnStore(N)` directly to construct a store.
"""
init_particles(N::Int) = ColumnStore(N)

"""
    ssm_step_body(a::Float64, q::Float64, r::Float64, data::Vector{Float64})

Per-iteration body factory for `build_ssm_filter`:

    x(t) ~ Normal(a * x(t-1), q)
    data[t] => Normal(x(t), r)

Returns a closure `t -> Sequence(Sample(...), Observe(...))`; `data[t]` is the
only thing that varies across iterations (a captured field value), so every
call returns the same concrete type.
"""
ssm_step_body(a::Float64, q::Float64, r::Float64, data::Vector{Float64}) =
    t -> Sequence(
        Sample(:x, NormalKernel, state -> (a .* getcol(state.store, :x), Ref(q))),
        Observe(state -> Ref(data[t]), NormalKernel, state -> (getcol(state.store, :x), Ref(r))),
    )

"""
    build_ssm_filter(data::Vector{Float64}, a::Float64, q::Float64, r::Float64)

Linear-Gaussian state-space model (no resampling, so this is plain importance
sampling over the whole sequence):

    x(0) ~ Normal(0, 1)
    x(t) ~ Normal(a * x(t-1), q)      for each observation
    data[t] => Normal(x(t), r)

Used to validate `Observe` against the exact Kalman filter solution.
"""
function build_ssm_filter(data::Vector{Float64}, a::Float64, q::Float64, r::Float64)
    init = Sample(:x, NormalKernel, state -> (Ref(0.0), Ref(1.0)))
    loop = Loop(state -> 1:length(data), ssm_step_body(a, q, r, data))
    return Sequence((init, loop))
end

"""
    ssm_step_body_weight(a::Float64, q::Float64, r::Float64, data::Vector{Float64})

Same per-iteration semantics as `ssm_step_body`, but implemented with `Weight`
instead of `Observe`: the observed value `data[t]` is folded directly into
`argfn`'s returned tuple (as a `Ref`) rather than kept in a separate `lhsfn`,
so `NormalKernel.logpdf(μ, σ, x)` is called with all 3 args coming from
`argfn`. Demonstrates that `Weight` subsumes `Observe` whenever the "observed"
value can just be treated as one more (typically constant) argument.
"""
ssm_step_body_weight(a::Float64, q::Float64, r::Float64, data::Vector{Float64}) =
    t -> Sequence(
        Sample(:x, NormalKernel, state -> (a .* getcol(state.store, :x), Ref(q))),
        Weight(NormalKernel, state -> (getcol(state.store, :x), Ref(r), Ref(data[t]))),
    )

"""
    build_ssm_filter_weight(data, a, q, r)

Same model as `build_ssm_filter`, but the observation step uses `Weight`
instead of `Observe` (see `ssm_step_body_weight`). Used to cross-check
`Weight` against the exact Kalman filter solution.
"""
function build_ssm_filter_weight(data::Vector{Float64}, a::Float64, q::Float64, r::Float64)
    init = Sample(:x, NormalKernel, state -> (Ref(0.0), Ref(1.0)))
    loop = Loop(state -> 1:length(data), ssm_step_body_weight(a, q, r, data))
    return Sequence((init, loop))
end

"""
    ssm_step_body_resampled(a::Float64, q::Float64, r::Float64, data::Vector{Float64})

Same as `ssm_step_body`, but with a `Resample` step appended after the
`Observe`, so particle diversity is maintained over long horizons.
"""
ssm_step_body_resampled(a::Float64, q::Float64, r::Float64, data::Vector{Float64}) =
    t -> Sequence(
        Sample(:x, NormalKernel, state -> (a .* getcol(state.store, :x), Ref(q))),
        Observe(state -> Ref(data[t]), NormalKernel, state -> (getcol(state.store, :x), Ref(r))),
        Resample(),
    )

"""
    build_ssm_filter_resampled(data, a, q, r)

Same model as `build_ssm_filter`, but resamples after every observation (via
`ssm_step_body_resampled`), so it stays stable over long horizons (unlike
`build_ssm_filter`, which is plain unresampled importance sampling and
degenerates for large `T`). Used to validate `Resample`/evidence-in-weights
against the exact Kalman filter over a longer `T`.
`ess_perc_min` is set on the `SMCState` passed to `apply!`, not here.
"""
function build_ssm_filter_resampled(data::Vector{Float64}, a::Float64, q::Float64, r::Float64)
    init = Sample(:x, NormalKernel, state -> (Ref(0.0), Ref(1.0)))
    loop = Loop(state -> 1:length(data), ssm_step_body_resampled(a, q, r, data))
    return Sequence((init, loop))
end

"""
    ssm_step_body_resampled_K(a, q, r, data, K)

`K`-variable generalization of `ssm_step_body_resampled`: `K` independent
state variables `x1, ..., xK`, each following the same linear-Gaussian
dynamics and observed against the SAME `data` series (statistically
redundant across `k` — this only grows the number of particle columns, for
benchmarking storage-backend overhead vs. variable count). A single
`Resample` is applied once per iteration, after all `K` observations, so it
combines the weight contributions from all `K` variables (matching how a
model with `K` real observed dimensions would resample jointly).

Every one of the `K` per-variable sub-`Sequence`s has the same concrete type
(`Sample{Symbol,...}`/`Observe{...}` don't encode the variable's name in
their type parameters), so the returned `Sequence`'s type does not grow with
`K` — only the runtime length of its `steps` tuple does.
"""
ssm_step_body_resampled_K(a::Float64, q::Float64, r::Float64, data::Vector{Float64}, K::Int) =
    t -> Sequence((
        ntuple(K) do k
            Sequence((
                Sample(Symbol(:x, k), NormalKernel, state -> (a .* getcol(state.store, Symbol(:x, k)), Ref(q))),
                Observe(state -> Ref(data[t]), NormalKernel, state -> (getcol(state.store, Symbol(:x, k)), Ref(r))),
            ))
        end...,
        Resample(),
    ))

"""
    build_ssm_filter_resampled_K(data, a, q, r, K)

`K`-variable generalization of `build_ssm_filter_resampled` (see
`ssm_step_body_resampled_K`): `K` independent state variables `x1, ..., xK`,
each initialized `xk(0) ~ Normal(0, 1)`, resampled jointly once per
observation. `K=1` reduces to the same model as `build_ssm_filter_resampled`
(just under the name `x1` instead of `x`).
"""
function build_ssm_filter_resampled_K(data::Vector{Float64}, a::Float64, q::Float64, r::Float64, K::Int)
    inits = ntuple(k -> Sample(Symbol(:x, k), NormalKernel, state -> (Ref(0.0), Ref(1.0))), K)
    loop = Loop(state -> 1:length(data), ssm_step_body_resampled_K(a, q, r, data, K))
    return Sequence((inits..., loop))
end

"""
    logsumexp(w)

`log(sum(exp.(w)))`, computed in a numerically stable way. Already provided
by `src/resampling.jl` (via `using WeightedSampling` above) — kept here only
as a docs pointer, not redefined.
"""

"""
    kalman_filter_evidence(data, a, q, r)

Exact Kalman filter for the 1D linear-Gaussian SSM `x(t) = a*x(t-1) + q*ε`,
`y(t) = x(t) + r*η`, `x(0) ~ Normal(0, 1)`. Returns `(posterior_mean_final,
log_evidence)`, used as the ground truth reference in
`transformers_test.jl`/`macro_test.jl`.
"""
function kalman_filter_evidence(data, a, q, r)
    μ, P = 0.0, 1.0
    log_evidence = 0.0
    for y in data
        μ_pred = a * μ
        P_pred = a^2 * P + q^2

        S = P_pred + r^2
        residual = y - μ_pred
        log_evidence += -0.5 * (log(2π) + log(S) + residual^2 / S)

        K = P_pred / S
        μ = μ_pred + K * residual
        P = (1 - K) * P_pred
    end
    return μ, log_evidence
end

