# Fused-broadcast invariant: every stochastic/weighting step writes its result
# with a single fused broadcast (`col .= f.(args...)` via `broadcast_setcol!`,
# or `weights .+= …`), never by materializing `f.(args...)` on its own first.
# The destination (length `N`) drives the iteration length, so even when every
# argument is `Ref`-wrapped (e.g. an i.i.d. draw `x ~ Normal(0, 1)`) the step
# produces `N` independent results instead of collapsing to a single scalar.

"""
    Assign{L,F} <: ParticleTransformer

Deterministic broadcast assignment `x .= expr`.

# Fields
- `lhs::L`: column name (`Symbol`) to assign into.
- `argfn::F`: closure `state -> value` computing the (already vectorized)
  right-hand side from the current `SMCState`.
"""
struct Assign{L,F} <: ParticleTransformer
    lhs::L
    argfn::F
end

"""
    apply!(t::Assign, state::SMCState)

Broadcast-assign `t.argfn(state)` into column `t.lhs`, creating it if needed.
"""
function apply!(t::Assign, state::SMCState)
    broadcast_setcol!(state.store, t.lhs, identity, (t.argfn(state),))
    advance!(state)
    return nothing
end

"""
    score!(t::Assign, state::SMCState, ctx::ScoreCtx)

No-op (an assignment is deterministic); advances `ctx.depth` to match `apply!`.
"""
function score!(t::Assign, state::SMCState, ctx::ScoreCtx)
    advance!(ctx)
    return nothing
end

"""
    AccessorAssign{F} <: ParticleTransformer

In-place mutation through a value-level accessor (`x[e] .= rhs`, `x.p .= rhs`,
and chains thereof, e.g. `x[e].p .= rhs`). Unlike `Assign`, this does not
replace a column: it calls `setindex!`/`setproperty!` on elements already
stored in an existing column, so the target column must already exist, and
(for `.p` writes) its elements must be a reference type (e.g. a mutable
struct) for the mutation to be visible through the store.

# Fields
- `fn::F`: closure `state -> nothing` performing the mutation as a side effect.
"""
struct AccessorAssign{F} <: ParticleTransformer
    fn::F
end

"""
    apply!(t::AccessorAssign, state::SMCState)

Run the mutation closure `t.fn(state)`, then advance the depth counter.
"""
function apply!(t::AccessorAssign, state::SMCState)
    t.fn(state)
    advance!(state)
    return nothing
end

"""
    score!(t::AccessorAssign, state::SMCState, ctx::ScoreCtx)

No-op (deterministic mutation); advances `ctx.depth` to match `apply!`.
"""
function score!(t::AccessorAssign, state::SMCState, ctx::ScoreCtx)
    advance!(ctx)
    return nothing
end

"""
    AccessorSample{RF,WF,K,F} <: ParticleTransformer

Stochastic sampling step through a value-level accessor target (`x[e] ~
f(args)`, `x.p ~ f(args)`, and chains thereof). Like `AccessorAssign`, this
does not replace a whole column: it draws one sample per particle and writes
each into an element of an existing column via `setindex!`/`setproperty!`, so
the target column must already exist (and, for `.p` writes, its elements must
be a reference type). Like `Sample`, it also adds a log-weight contribution
when `kernel.weighter !== nothing`.

# Fields
- `readfn::RF`: closure `state -> values` reading the current element values
  through the accessor (used by `score!`).
- `writefn::WF`: closure `(state, values) -> nothing` writing freshly sampled
  `values` into the accessor's elements.
- `kernel::K`: a `WeightedKernel` providing `sampler`, (optionally)
  `weighter`, and `logpdf`.
- `argfn::F`: closure `state -> args_tuple` computing the (already vectorized)
  kernel arguments.
"""
struct AccessorSample{RF,WF,K,F} <: ParticleTransformer
    readfn::RF
    writefn::WF
    kernel::K
    argfn::F
end

"""
    apply!(t::AccessorSample, state::SMCState)

Draw one sample per particle, write it into the target elements via
`t.writefn`, and (if `t.kernel.weighter !== nothing`) add the weighter's
contribution to the log-weights. The sample is drawn as a fused broadcast into
a fresh length-`N` vector (see the fused-broadcast invariant above).
"""
function apply!(t::AccessorSample, state::SMCState)
    args = t.argfn(state)
    n = nparticles(state.store)
    ET = Broadcast.combine_eltypes(t.kernel.sampler, args)
    values = Vector{ET}(undef, n)
    broadcast!(t.kernel.sampler, values, args...)
    t.writefn(state, values)
    if t.kernel.weighter !== nothing
        state.weights .+= t.kernel.weighter.(args..., values)
        state.weights_changed = true
    end
    advance!(state)
    return nothing
end

"""
    score!(t::AccessorSample, state::SMCState, ctx::ScoreCtx)

Add `t.kernel.logpdf.(args..., x)` to `ctx.scores`, where `x = t.readfn(state)`
is the current value of the target elements (read, not re-sampled).
"""
function score!(t::AccessorSample, state::SMCState, ctx::ScoreCtx)
    args = t.argfn(state)
    x = t.readfn(state)
    ctx.scores .+= t.kernel.logpdf.(args..., x)
    advance!(ctx)
    return nothing
end

"""
    Sample{L,K,F} <: ParticleTransformer

Stochastic sampling step `x ~ f(args)`, where `f` is a `WeightedKernel`.

# Fields
- `lhs::L`: column name (`Symbol`) to sample into.
- `kernel::K`: a `WeightedKernel` providing `sampler` and (optionally) `weighter`.
- `argfn::F`: closure `state -> args_tuple` computing the (already vectorized)
  kernel arguments from the current `SMCState`.
"""
struct Sample{L,K,F} <: ParticleTransformer
    lhs::L
    kernel::K
    argfn::F
end

"""
    apply!(t::Sample, state::SMCState)

Draw `col .= t.kernel.sampler.(args...)` into column `t.lhs` (a fused
broadcast, see the fused-broadcast invariant above), where `args =
t.argfn(state)`. If `t.kernel.weighter !== nothing`, add
`t.kernel.weighter.(args..., values)` to the log-weights.
"""
function apply!(t::Sample, state::SMCState)
    args = t.argfn(state)
    broadcast_setcol!(state.store, t.lhs, t.kernel.sampler, args)
    if t.kernel.weighter !== nothing
        values = getcol(state.store, t.lhs)
        state.weights .+= t.kernel.weighter.(args..., values)
        state.weights_changed = true
    end
    advance!(state)
    return nothing
end

"""
    score!(t::Sample, state::SMCState, ctx::ScoreCtx)

Add `t.kernel.logpdf.(args..., x)` to `ctx.scores`, where `x` is the current
value of column `t.lhs` (read from `state.store`, not re-sampled). Reading the
current value — rather than re-drawing — is what makes `score!` cheap and is
correct for a `Move` provided every factor whose density depends on the move
targets still holds the value it had when the statement originally executed.
"""
function score!(t::Sample, state::SMCState, ctx::ScoreCtx)
    args = t.argfn(state)
    x = getcol(state.store, t.lhs)
    ctx.scores .+= t.kernel.logpdf.(args..., x)
    advance!(ctx)
    return nothing
end

"""
    Observe{V,K,F} <: ParticleTransformer

Conditioning/observation step `expr => f(args)`, where `f` is a
`WeightedKernel`. Adds `f.logpdf.(args..., expr)` to the log-weights; does not
sample or store anything.

# Fields
- `lhsfn::V`: closure `state -> value` computing the (already vectorized)
  observed value `expr` (typically `Ref(y)` for a fixed data point, since
  observations are usually external data, not particle-dependent).
- `kernel::K`: a `WeightedKernel` providing `logpdf`.
- `argfn::F`: closure `state -> args_tuple` computing the (already vectorized)
  kernel arguments.
"""
struct Observe{V,K,F} <: ParticleTransformer
    lhsfn::V
    kernel::K
    argfn::F
end

"""
    apply!(t::Observe, state::SMCState)

Add `t.kernel.logpdf.(args..., t.lhsfn(state))` to the log-weights (a fused
`.+=` broadcast), where `args = t.argfn(state)`.
"""
function apply!(t::Observe, state::SMCState)
    args = t.argfn(state)
    obs = t.lhsfn(state)
    state.weights .+= t.kernel.logpdf.(args..., obs)
    state.weights_changed = true
    advance!(state)
    return nothing
end

"""
    score!(t::Observe, state::SMCState, ctx::ScoreCtx)

Add `t.kernel.logpdf.(args..., obs)` to `ctx.scores` — the same contribution
`apply!` added to the weights.
"""
function score!(t::Observe, state::SMCState, ctx::ScoreCtx)
    args = t.argfn(state)
    obs = t.lhsfn(state)
    ctx.scores .+= t.kernel.logpdf.(args..., obs)
    advance!(ctx)
    return nothing
end

"""
    Weight{K,F} <: ParticleTransformer

Weighting-without-sampling step `_ ~ f(args)`: adds a log-weight contribution
to every particle without drawing or storing a sample. Reuses `WeightedKernel`
for consistency with `Sample`/`Observe`, but `kernel.logpdf` here has a
different arity: `logpdf(args...) -> logweight` (no trailing sample argument).
`sampler`/`weighter` are unused (typically `nothing`).

# Fields
- `kernel::K`: a `WeightedKernel` whose `logpdf` is `(args...) -> logweight`.
- `argfn::F`: closure `state -> args_tuple` computing the (already vectorized)
  kernel arguments.
"""
struct Weight{K,F} <: ParticleTransformer
    kernel::K
    argfn::F
end

"""
    apply!(t::Weight, state::SMCState)

Add `t.kernel.logpdf.(args...)` to the log-weights (a fused `.+=` broadcast),
where `args = t.argfn(state)`.
"""
function apply!(t::Weight, state::SMCState)
    args = t.argfn(state)
    state.weights .+= t.kernel.logpdf.(args...)
    state.weights_changed = true
    advance!(state)
    return nothing
end

"""
    score!(t::Weight, state::SMCState, ctx::ScoreCtx)

Add `t.kernel.logpdf.(args...)` to `ctx.scores`, mirroring `apply!`.
"""
function score!(t::Weight, state::SMCState, ctx::ScoreCtx)
    args = t.argfn(state)
    ctx.scores .+= t.kernel.logpdf.(args...)
    advance!(ctx)
    return nothing
end


"""
    Sequence{Ts<:Tuple} <: ParticleTransformer

Composite transformer: applies a fixed `Tuple` of steps in order. Since the
steps are stored as a `Tuple` (not a `Vector`), the tuple's heterogeneous
element types are known at compile time, so `apply!` recurses over `t.steps`
and the whole chain can be specialized/inlined rather than dynamically
dispatched per step.

# Constructors
```julia
Sequence(steps...)   # e.g. Sequence(assign1, sample1, sample2)
Sequence(steps_tuple)
```
"""
struct Sequence{Ts<:Tuple} <: ParticleTransformer
    steps::Ts
end

Sequence(steps...) = Sequence(steps)

"""
    apply!(t::Sequence, state::SMCState)

Apply each step in `t.steps` to `state`, in order.
"""
function apply!(t::Sequence, state::SMCState)
    foreach(step -> apply!(step, state), t.steps)
    return nothing
end

"""
    score!(t::Sequence, state::SMCState, ctx::ScoreCtx)

Recurse over `t.steps` in order, stopping as soon as
`ctx.depth >= ctx.target_depth` (i.e. once every statement executed strictly
before the move has been scored).
"""
function score!(t::Sequence, state::SMCState, ctx::ScoreCtx)
    for step in t.steps
        ctx.depth < ctx.target_depth || break
        score!(step, state, ctx)
    end
    return nothing
end

"""
    Loop{F,B} <: ParticleTransformer

Represents `for x in collection; body; end`, without unrolling `collection`
into a flat `Sequence` at construction time (unrolling would force a fresh
compiled specialization per program length).

# Fields
- `collfn::F`: closure `state -> iterable` computing the loop's collection
  (must not depend on particle variables).
- `bodyfn::B`: closure `x -> ParticleTransformer`, called once per loop element
  to build that iteration's body. Every call returns the same concrete type
  (only captured values differ), so `apply!` still compiles to a single
  specialization regardless of the iteration count; only a small transformer
  object is (re)allocated per iteration.
"""
struct Loop{F,B} <: ParticleTransformer
    collfn::F
    bodyfn::B
end

"""
    apply!(t::Loop, state::SMCState)

For each `x` in `t.collfn(state)`, build `t.bodyfn(x)` and apply it to
`state`.
"""
function apply!(t::Loop, state::SMCState)
    for x in t.collfn(state)
        apply!(t.bodyfn(x), state)
    end
    return nothing
end

"""
    score!(t::Loop, state::SMCState, ctx::ScoreCtx)

For each `x` in `t.collfn(state)`, rebuild `t.bodyfn(x)` (the same loop value
the forward pass used, so loop-variable-dependent arguments reconstruct
correctly) and score it, early-terminating once `ctx.depth >= ctx.target_depth`.
"""
function score!(t::Loop, state::SMCState, ctx::ScoreCtx)
    for x in t.collfn(state)
        ctx.depth < ctx.target_depth || break
        score!(t.bodyfn(x), state, ctx)
    end
    return nothing
end

"""
    Cond{P,B} <: ParticleTransformer

Represents `if condition; body; end` (no `else` branch). Unlike `Loop`, a
conditional does not repeat, so its `body` is a single fixed transformer built
once.

# Fields
- `predfn::P`: closure `state -> Bool` evaluating the condition. Must not
  depend on particle variables; it may reference `state` fields such as
  `state.resampled` (e.g. `if resampled ... end`) and captured locals.
- `body::B`: the `ParticleTransformer` to run when `predfn(state)` is `true`.
"""
struct Cond{P,B} <: ParticleTransformer
    predfn::P
    body::B
end

"""
    apply!(t::Cond, state::SMCState)

Apply `t.body` to `state` iff `t.predfn(state)` is `true`; otherwise a no-op.
"""
function apply!(t::Cond, state::SMCState)
    if t.predfn(state)
        apply!(t.body, state)
    end
    return nothing
end

"""
    score!(t::Cond, state::SMCState, ctx::ScoreCtx)

Re-evaluate `t.predfn(state)` and score `t.body` iff it holds. Since `predfn`
may read transient state (e.g. `state.resampled`), it can pick a different
branch than the forward pass took; this is only safe when the gated body
contains nothing but depth-/score-neutral `Move`s (the idiomatic
`if resampled; x << q(); end` pattern). Not enforced.
"""
function score!(t::Cond, state::SMCState, ctx::ScoreCtx)
    if t.predfn(state)
        score!(t.body, state, ctx)
    end
    return nothing
end

"""
    Resample <: ParticleTransformer

Resampling step: resamples particles proportional to their (unnormalized)
cumulative log-weights whenever `state.weights_changed` is set and the
resulting ESS% falls below `state.ess_perc_min`; otherwise a no-op (besides
clearing `weights_changed`).

Does not reset weights to a uniform value. Instead every particle's log-weight
(resampled or not) is reset to the current log-mean `logsumexp(weights) -
log(N)`. Since resampling draws indices proportional to the weights, the mean
of the resampled weights equals the pre-resample mean, so
`logsumexp(weights) - log(N)` (the evidence) stays correct across resampling
without a separate accumulator.
"""
struct Resample <: ParticleTransformer end

"""
    apply!(t::Resample, state::SMCState)

No-op (leaving `state.resampled` unchanged) if `state.weights_changed` is
`false` — nothing has been weighted since the last resampling decision, so an
auto-inserted `Resample` must not overwrite the outcome of the previous one
(important for `if resampled` after several auto-inserted resamples).
Otherwise: normalize weights, compute ESS%, and resample (stratified) if
ESS% < `state.ess_perc_min`, resetting every particle's log-weight to the
pre-resample log-mean. Always clears `state.weights_changed` afterwards.
"""
function apply!(t::Resample, state::SMCState)
    if !state.weights_changed
        return nothing
    end

    logW = state.weights
    w = exp_norm(logW)

    current_ess_perc = ess_perc(w)

    if current_ess_perc < state.ess_perc_min
        indices = stratified_resample(w)
        mean_logW = logsumexp(logW) - log(length(logW))

        resample!(state.store, indices)
        fill!(state.weights, mean_logW)

        state.resampled = true
    else
        state.resampled = false
    end

    state.weights_changed = false
    return nothing
end

"""
    score!(t::Resample, state::SMCState, ctx::ScoreCtx)

No-op: resampling is not a modeled random choice and is depth-neutral.
"""
function score!(t::Resample, state::SMCState, ctx::ScoreCtx)
    return nothing
end

"""
    Move{T,Q,F} <: ParticleTransformer

Metropolis–Hastings move `x << q(args)`: proposes new values for the target
column(s) and accepts/rejects per particle so the move leaves the current SMC
target distribution invariant.

# Fields
- `targets::T`: `Vector{Symbol}` of column(s) being moved.
- `proposal::Q`: a proposal function
  `proposal(state, targets, args...) -> (proposed::Dict{Symbol,<:AbstractVector}, log_pratio::Vector{Float64})`,
  where `log_pratio` is the log proposal ratio
  `log q(old|new) - log q(new|old)` (zero for a symmetric proposal like `RW`).
- `argfn::F`: closure `state -> args_tuple` computing the (unvectorized)
  proposal arguments, e.g. a step size.

Depth- and score-neutral: `score!` is a no-op and `apply!` does not advance
`state.depth`. A move contributes nothing to the trace density (it only
reshuffles the target coordinate given the latest randomness), and staying
depth-neutral keeps `if resampled; x << q(); end` robust under re-scoring.
"""
struct Move{T,Q,F} <: ParticleTransformer
    targets::T
    proposal::Q
    argfn::F
end

"""
    apply!(t::Move, state::SMCState)

Run one Metropolis–Hastings step:
1. Capture `depth = state.depth` (before anything, since `Move` is
   depth-neutral) and propose new target values via `t.proposal`.
2. Score the trace log-density at the old values into `state.score_buf1`
   (`score_logpdf!` folds `score!` over `state.root`, which must therefore be
   set — see [`run!`](@ref)).
3. Write the proposed values through `broadcast_setcol!`, saving the old
   values for possible rejection.
4. Score the trace log-density at the new values into `state.score_buf2`.
5. Accept/reject per particle; rejected particles have their target column(s)
   restored from the saved old values.

Weights are untouched: an MH move re-parameterizes particles within the
current target and must not change importance weights or the evidence.
"""
function apply!(t::Move, state::SMCState)
    store = state.store
    targets = t.targets
    depth = state.depth
    args = t.argfn(state)

    proposed, log_pratio = t.proposal(state, targets, args...)

    s_old = score_logpdf!(state.score_buf1, state, targets, depth)

    cols = map(c -> getcol(store, c), targets)
    old = map(copy, cols)
    for (k, c) in enumerate(targets)
        broadcast_setcol!(store, c, identity, (proposed[c],))
    end
    # broadcast_setcol! mutates existing columns in place, but re-fetch to be
    # robust against any backend that could reallocate a column buffer.
    cols = map(c -> getcol(store, c), targets)

    s_new = score_logpdf!(state.score_buf2, state, targets, depth)

    @inbounds for i in eachindex(log_pratio)
        if !(log(rand()) < log_pratio[i] + s_new[i] - s_old[i])
            for k in eachindex(targets)
                cols[k][i] = old[k][i]
            end
        end
    end

    return nothing
end

"""
    score!(t::Move, state::SMCState, ctx::ScoreCtx)

No-op: a move already reflects the latest random choices and contributes
nothing to the trace log-density. Does not advance `ctx.depth`.
"""
function score!(t::Move, state::SMCState, ctx::ScoreCtx)
    return nothing
end

# --- Pretty printing ---------------------------------------------------
#
# `ParticleTransformer`s carry closures (`argfn`, `kernel`, `writefn`, ...)
# as type parameters, so Julia's default struct printing recurses into
# enormous, unreadable closure type signatures. Instead we print a short
# label per node (step kind + any identifying `Symbol`(s), e.g. the target
# column) and, for container nodes whose children are known statically
# (`Sequence`, `Cond`), an indented tree of those children.
#
# `Loop`'s body is NOT shown: it's built by calling `bodyfn(x)` fresh for
# each loop element, so there is no single body to display statically.

_label(t::Assign) = "Assign(:$(t.lhs))"
_label(t::AccessorAssign) = "AccessorAssign"
_label(t::AccessorSample) = "AccessorSample"
_label(t::Sample) = "Sample(:$(t.lhs))"
_label(t::Observe) = "Observe"
_label(t::Weight) = "Weight"
_label(t::Sequence) = "Sequence"
_label(t::Loop) = "Loop"
_label(t::Cond) = "Cond"
_label(t::Resample) = "Resample()"
_label(t::Move) = "Move($(t.targets))"

_children(t::ParticleTransformer) = ParticleTransformer[]
_children(t::Sequence) = collect(ParticleTransformer, t.steps)
_children(t::Cond) = ParticleTransformer[t.body]

function _print_tree(io::IO, t::ParticleTransformer, prefix::AbstractString)
    print(io, _label(t))
    t isa Loop && print(io, " (body built per-iteration, not shown)")
    kids = _children(t)
    for (i, kid) in enumerate(kids)
        islast = i == length(kids)
        print(io, "\n", prefix, islast ? "└─ " : "├─ ")
        _print_tree(io, kid, prefix * (islast ? "   " : "│  "))
    end
end

"""
    show(io, x::ParticleTransformer)
    show(io, ::MIME"text/plain", x::ParticleTransformer)

Compact one-line label (2-arg `show`, e.g. `Sample(:α)`) or an indented tree
view (`MIME"text/plain"`, used by the REPL) of a built model, e.g.:

```
Sequence
├─ Sample(:α)
├─ Sample(:β)
└─ Loop (body built per-iteration, not shown)
```
"""
Base.show(io::IO, t::ParticleTransformer) = print(io, _label(t))
Base.show(io::IO, ::MIME"text/plain", t::ParticleTransformer) = _print_tree(io, t, "")
