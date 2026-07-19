"""
    ParticleTransformer

Base type for all particle-transforming operations. Every operation on the
particles (assignment, sampling, weighting, ...) is a `ParticleTransformer`,
and they compose via [`Sequence`](@ref).
"""
abstract type ParticleTransformer end

"""
    SMCState(store::AbstractParticleStore; rng=Random.default_rng(), ess_perc_min=0.5, show_progress=false)

Mutable container for everything an SMC run needs, threaded through every
`apply!` call.

# Fields
- `store`: an `AbstractParticleStore` (e.g. `ColumnStore`) holding the
  particle variable columns. All column access goes through the store
  interface, so the backend is swappable.
- `weights::Vector{Float64}`: cumulative *unnormalized* log-weights, kept as a
  concrete `Vector{Float64}` (separate from the variable columns) so the
  hottest broadcast `weights .+= …` is type-stable. The evidence is always
  `logsumexp(weights) - log(N)`; `Resample` preserves this quantity.
- `rng`: currently unused by kernels (they call the global `rand()`
  internally); reserved for future per-thread RNG threading.
- `resampled::Bool`: whether the most recent `Resample` step actually
  resampled. Visible to model bodies via `if resampled ... end`.
- `weights_changed::Bool`: set by weighting steps (`Sample` with a `weighter`,
  `Observe`, `Weight`); consumed and reset by `Resample`, which only resamples
  if it is set.
- `ess_perc_min::Float64`: minimum effective-sample-size percentage below
  which `Resample` triggers stratified resampling.
- `depth::Int`: dynamic execution-depth counter, incremented by one for every
  executed `Assign`/`Sample`/`Weight`/`Observe` step (in both `apply!` and
  `score!`) so a depth captured during a forward run names the same program
  point during a later `score!` walk. `Move`/`Resample` do not advance it.
  Starts at `0`; reset by `run!`.
- `root`: the top-level `ParticleTransformer`, set by `run!`, so a `Move`
  anywhere in the tree can re-fold `score!` over the whole program. `nothing`
  until `run!` is called (fine for any program with no `Move`).
- `progress`: a `ProgressMeter.ProgressUnknown` ticked once per executed
  counted step. Controlled by the `show_progress` keyword (default `false`);
  `run!` finishes it when the top-level transformer returns.
- `score_buf1`, `score_buf2`: reusable per-particle log-density scratch buffers
  used by `Move.apply!` to avoid allocating a fresh `zeros(N)` per step. Start
  empty and grow to length `N` on first use.
"""
mutable struct SMCState{S,R}
    store::S
    weights::Vector{Float64}
    rng::R
    resampled::Bool
    weights_changed::Bool
    ess_perc_min::Float64
    depth::Int
    root::Any
    progress::ProgressUnknown
    score_buf1::Vector{Float64}
    score_buf2::Vector{Float64}
end

SMCState(store::AbstractParticleStore; rng=Random.default_rng(), ess_perc_min=0.5, show_progress=false) =
    SMCState(store, zeros(nparticles(store)), rng, false, false, ess_perc_min, 0, nothing,
        ProgressUnknown(desc="Steps performed:", dt=0.1, showspeed=true, color=:blue, enabled=show_progress),
        Float64[], Float64[])

"""
    SMCState(n_particles::Integer; rng=Random.default_rng(), ess_perc_min=0.5, show_progress=false)

Convenience constructor: builds a `ColumnStore(n_particles)` internally, so
callers don't need to interact with `ColumnStore` directly.

# Examples
```julia
state = SMCState(1000)
```
"""
SMCState(n_particles::Integer; kwargs...) = SMCState(ColumnStore(n_particles); kwargs...)

"""
    getindex(state::SMCState, name::Symbol) -> AbstractVector

Read-only access to a particle variable column by name, e.g. `state[:x]`.
There is no corresponding `setindex!` — the only write path into the store is
`broadcast_setcol!`.
"""
Base.getindex(state::SMCState, name::Symbol) = getcol(state.store, name)

"""
    show(io, state::SMCState)

Compact summary of the state. The 2-arg form prints a single line
(`SMCState(n_particles=1000, columns=[:α, :β])`); the `MIME"text/plain"` form
(used by the REPL) additionally lists `ess_perc_min`/`resampled`/`depth`.
"""
function Base.show(io::IO, state::SMCState)
    print(io, "SMCState(n_particles=", nparticles(state.store),
        ", columns=", colnames(state.store), ")")
end

function Base.show(io::IO, ::MIME"text/plain", state::SMCState)
    println(io, "SMCState")
    println(io, "  n_particles:  ", nparticles(state.store))
    println(io, "  columns:      ", colnames(state.store))
    println(io, "  ess_perc_min: ", state.ess_perc_min)
    println(io, "  resampled:    ", state.resampled)
    print(io, "  depth:        ", state.depth)
end

"""
    run!(root::ParticleTransformer, state::SMCState)

Top-level entry point for running a program. Sets `state.root = root` (so a
`Move` anywhere in the tree can re-fold `score!` over the whole program),
resets `state.depth = 0`, applies `root` to `state`, and finishes the progress
meter.

Programs with no `Move` steps can instead call `apply!(model, state)` directly.
"""
function run!(root::ParticleTransformer, state::SMCState)
    state.root = root
    state.depth = 0
    apply!(root, state)
    finish!(state.progress)
    return state
end

"""
    ScoreCtx(targets, target_depth, depth, scores)

Mutable accumulator threaded through a `score!` walk — the density dual of
`SMCState` for `apply!`. `score!` mutates it in place rather than returning a
value, so the walk composes as a fold with early termination.

# Fields
- `targets::Vector{Symbol}`: the column(s) being moved. Currently threaded
  through unused (the present version scores every factor); reserved for a
  future taint-based pruning optimization.
- `target_depth::Int`: cutoff. Only statements executed strictly before the
  move (`depth < target_depth`) are scored; the walk early-terminates at this
  depth, which also prevents `score!` from reading columns that don't exist
  yet mid-loop.
- `depth::Int`: running execution-depth counter, incremented by the same rule
  as `SMCState.depth`.
- `scores::Vector{Float64}`: per-particle log-density accumulator (length `N`).
"""
mutable struct ScoreCtx
    targets::Vector{Symbol}
    target_depth::Int
    depth::Int
    scores::Vector{Float64}
end

"""
    advance!(state::SMCState)

Advance the per-step bookkeeping for one executed counted step: increment the
execution-depth counter and tick the progress meter (gated on
`state.progress.enabled`). Centralizes what would otherwise be repeated in
every counted `apply!` method, keeping the forward and score passes in sync.
"""
@inline function advance!(state::SMCState)
    state.depth += 1
    state.progress.enabled && next!(state.progress)
    return nothing
end

"""
    advance!(ctx::ScoreCtx)

Score-pass counterpart of [`advance!(::SMCState)`](@ref): increment the
`score!`-walk depth counter by one, matching the forward pass.
"""
@inline function advance!(ctx::ScoreCtx)
    ctx.depth += 1
    return nothing
end

"""
    score_logpdf(state::SMCState, targets, target_depth) -> Vector{Float64}

Compute the per-particle trace log-density of `state.root` up to (but not
including) `target_depth`, by folding `score!` over the whole program tree.
Requires `state.root` to have been set (via [`run!`](@ref)). Allocates a fresh
result vector; use the in-place [`score_logpdf!`](@ref) to reuse a buffer.
"""
function score_logpdf(state::SMCState, targets, target_depth::Int)
    return score_logpdf!(zeros(nparticles(state.store)), state, targets, target_depth)
end

"""
    score_logpdf!(scores::Vector{Float64}, state::SMCState, targets, target_depth) -> scores

In-place variant of [`score_logpdf`](@ref): zeroes and reuses the caller-owned
`scores` buffer (resizing to `nparticles` if needed) instead of allocating a
fresh one, then folds `score!` over `state.root`.
"""
function score_logpdf!(scores::Vector{Float64}, state::SMCState, targets, target_depth::Int)
    n = nparticles(state.store)
    length(scores) == n || resize!(scores, n)
    fill!(scores, 0.0)
    ctx = ScoreCtx(targets isa Vector{Symbol} ? targets : collect(Symbol, targets),
        target_depth, 0, scores)
    score!(state.root, state, ctx)
    return scores
end

"""
    WeightedKernel{S,W,L}

A (randomly weighted) importance sampler. Operates on scalars;
`ParticleTransformer`s (e.g. `Sample`) broadcast it over columns.

# Fields
- `sampler::S`: `(args...) -> sample`, generates a sample.
- `weighter::W`: `(args..., sample) -> log_weight`, computes the log weight
  (may be random). Use `nothing` for uniform weights.
- `logpdf::L`: `(args..., sample) -> log_density`, evaluates the kernel's
  log-density.

# Constructor
```julia
WeightedKernel(sampler, weighter, logpdf)
```
"""
struct WeightedKernel{S,W,L}
    sampler::S
    weighter::W
    logpdf::L
end
