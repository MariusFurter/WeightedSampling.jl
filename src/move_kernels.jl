"""
Metropolis–Hastings proposal kernels for the `Move` transformer. Each takes a
`state` (for store and weight access) and returns a `Dict{Symbol,Vector{Float64}}`
of proposed columns, which `Move.apply!` writes through `broadcast_setcol!`.

# Proposal interface
    proposal(state, targets, args...) -> (proposed::Dict{Symbol,<:AbstractVector},
                                           log_pratio::Vector{Float64})

`log_pratio` is the log proposal ratio `log q(old|new) - log q(new|old)`
(zero for a symmetric proposal like both of these).
"""

"""
    _normalize_bounds(bounds, d)

Normalize the `bounds` argument of `RW`/`autoRW` into a `Vector` of length `d`
of `(lo, hi)` tuples. `nothing` becomes `d` copies of `(-Inf, Inf)` (fully
unconstrained, the default); a single `(lo, hi)` tuple is broadcast to all `d`
targets; a `Vector` of tuples is used as-is (must have length `d`). Use
`-Inf`/`Inf` for one-sided or unconstrained dimensions.
"""
function _normalize_bounds(bounds, d)
    bounds === nothing && return fill((-Inf, Inf), d)
    bounds isa Tuple && return fill(bounds, d)
    length(bounds) == d || throw(ArgumentError("bounds must have length $d (one (lo, hi) tuple per target), got $(length(bounds))"))
    return collect(bounds)
end

"""
    _to_unconstrained(x, lo, hi)

Map a constrained value `x ∈ (lo, hi)` to an unconstrained `z ∈ ℝ`: a logit
transform if both bounds are finite, a log transform if only one is finite,
or the identity if unconstrained.
"""
function _to_unconstrained(x, lo, hi)
    if isfinite(lo) && isfinite(hi)
        return log(x - lo) - log(hi - x)
    elseif isfinite(lo)
        return log(x - lo)
    elseif isfinite(hi)
        return log(hi - x)
    else
        return x
    end
end

"""
    _from_unconstrained(z, lo, hi)

Inverse of `_to_unconstrained`: map `z ∈ ℝ` back to `x ∈ (lo, hi)`.
"""
function _from_unconstrained(z, lo, hi)
    if isfinite(lo) && isfinite(hi)
        return lo + (hi - lo) / (1 + exp(-z))
    elseif isfinite(lo)
        return lo + exp(z)
    elseif isfinite(hi)
        return hi - exp(z)
    else
        return z
    end
end

_log1pexp(z) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))

"""
    _log_abs_jacobian(z, lo, hi)

Log absolute derivative `log|dx/dz|` of `_from_unconstrained` at `z`. Needed
to correct the Metropolis-Hastings proposal ratio when a symmetric random
walk step in unconstrained `z`-space is mapped back to constrained `x`-space:
the induced proposal on `x` is generally asymmetric, with
`log q(x|x') - log q(x'|x) = log|dx/dz|(z') - log|dx/dz|(z)`.
"""
function _log_abs_jacobian(z, lo, hi)
    if isfinite(lo) && isfinite(hi)
        return log(hi - lo) - _log1pexp(z) - _log1pexp(-z)
    elseif isfinite(lo) || isfinite(hi)
        return z
    else
        return 0.0
    end
end

# ---------------------------------------------------------------------------
# Shared proposal internals (RW and autoRW differ only in how they generate
# the per-target increments `changes`; everything else — argument unwrapping,
# building the unconstrained current-value matrix, and mapping the proposal
# back to constrained space with the Jacobian correction — is identical).
# ---------------------------------------------------------------------------

"""
    _scalar_arg(x)

Unwrap a possibly array-wrapped scalar proposal argument (e.g. a step size /
min step that arrived as a length-1 vector) down to a plain scalar; pass
scalars through unchanged.
"""
_scalar_arg(x) = x isa AbstractArray && !isempty(x) ? x[1] : x

"""
    _unconstrained_matrix(store, targets, bnds) -> Matrix

Stack the targets' current column values, each mapped to unconstrained space
via `_to_unconstrained`, into an `N×d` matrix (one column per target).

`hcat(cols...)` (not `reduce(hcat, cols)`) is required: with a single target
(`d == 1`), `reduce` short-circuits on a one-element collection and returns
that element UNCHANGED (a plain `Vector`, never calling `hcat`), so `cov`
would then return a scalar instead of a 1×1 `Matrix` — breaking the
`Σ[Σ.==0.0] .= min_step` broadcast in `autoRW`. `hcat` called directly (even
with one argument) always reshapes into an `N×1` `Matrix`.
"""
function _unconstrained_matrix(store, targets, bnds)
    return hcat((_to_unconstrained.(getcol(store, c), bnds[i][1], bnds[i][2])
                 for (i, c) in enumerate(targets))...)
end

"""
    _column_matrix(store, targets) -> Matrix

Stack the targets' current column values (untransformed) into an `N×d` matrix,
one column per target. The unbounded (`bounds === nothing`) counterpart of
[`_unconstrained_matrix`](@ref): since the transform is the identity there,
this skips the per-target `_to_unconstrained` broadcast copies. Uses `hcat`
(not `reduce(hcat, …)`) for the single-target `d == 1` reason documented on
`_unconstrained_matrix`.
"""
_column_matrix(store, targets) = hcat((getcol(store, c) for c in targets)...)

"""
    _adaptive_changes(state, z_old, min_step, d) -> Matrix

The shared adaptive core of `autoRW` (bounded and unbounded alike): from the
current (possibly unconstrained) target values `z_old` (`N×d`), form the
weighted empirical covariance `Σ` (weights `exp_norm(state.weights)`), replace
exactly-zero entries with `min_step` to keep `Σ` non-singular, and draw the
per-target proposal increments `changes = rand(MvNormal(λΣ), N)` (`d×N`),
`λ = 2.38 d^(-1/2)`. Kept in one place so the covariance/scaling logic can't
drift between the two paths.
"""
function _adaptive_changes(state::SMCState, z_old, min_step, d)
    N = size(z_old, 1)
    λ = 2.38 * d^(-1 / 2)
    w = ProbabilityWeights(exp_norm(state.weights))
    Σ = cov(z_old, w)
    Σ[Σ.==0.0] .= min_step
    return rand(MvNormal(λ * Σ), N)
end

"""
    _finish_proposal(z_old, changes, targets, bnds) -> (proposed::Dict, log_pratio::Vector)

Shared tail of `RW`/`autoRW`: given the unconstrained current values `z_old`
(`N×d`) and per-target proposal increments `changes` (`d×N`), add the
increments in unconstrained space, map each target back to constrained space
via `_from_unconstrained`, and accumulate the Jacobian log proposal ratio.
"""
function _finish_proposal(z_old, changes, targets, bnds)
    N = size(z_old, 1)
    proposed = Dict{Symbol,Vector{Float64}}()
    log_pratio = zeros(N)
    for (i, c) in enumerate(targets)
        lo, hi = bnds[i]
        zo = @view z_old[:, i]
        zn = zo .+ @view changes[i, :]
        proposed[c] = _from_unconstrained.(zn, lo, hi)
        log_pratio .+= _log_abs_jacobian.(zn, lo, hi) .- _log_abs_jacobian.(zo, lo, hi)
    end
    return proposed, log_pratio
end

"""
    RW(state, targets, step_size, bounds=nothing)

Symmetric random-walk proposal with variance `step_size` (isotropic:
`MvNormal(zeros(d), step_size)`).

When `bounds` is `nothing` (default), the proposal is symmetric and
`log_pratio` is all-zero. When `bounds` gives interval constraints (a single
`(lo, hi)` tuple applied to all targets, or a `Vector` of `(lo, hi)` tuples
matching `targets`, using `-Inf`/`Inf` for one-sided/unconstrained
dimensions), the walk is performed in an unconstrained transformed space so
proposals automatically satisfy the constraints; `log_pratio` then carries the
resulting Jacobian correction. The target log-density is unaffected either way.
"""
function RW(state::SMCState, targets, step_size, bounds=nothing)
    store = state.store
    N = nparticles(store)
    step = _scalar_arg(step_size)

    if bounds === nothing
        # Unconstrained fast path (the common case): a symmetric isotropic
        # random walk taken directly in the model's own coordinates. The
        # transform is the identity, so there is no `z` matrix and no
        # Jacobian, and the proposal ratio is identically zero.
        proposed = Dict{Symbol,Vector{Float64}}()
        for c in targets
            proposed[c] = getcol(store, c) .+ step .* randn(N)
        end
        return proposed, zeros(N)
    end

    d = length(targets)
    bnds = _normalize_bounds(bounds, d)
    z_old = _unconstrained_matrix(store, targets, bnds)
    changes = rand(MvNormal(zeros(d), step^2 * I), N)

    return _finish_proposal(z_old, changes, targets, bnds)
end

"""
    autoRW(state, targets, min_step=1e-3, bounds=nothing)

Adaptive random-walk proposal with covariance `λΣ`, where `λ = 2.38 d^(-1/2)`
(the standard optimal-scaling factor) and `Σ` is the weighted empirical
covariance of the target columns (weights = `exp_norm(state.weights)`).
Covariance entries that are exactly zero are replaced with `min_step` to avoid
a singular covariance.

When `bounds` is `nothing` (default), the proposal is symmetric and
`log_pratio` is all-zero. When `bounds` gives interval constraints (a single
`(lo, hi)` tuple applied to all targets, or a `Vector` of `(lo, hi)` tuples
matching `targets`, using `-Inf`/`Inf` for one-sided/unconstrained
dimensions), the walk (and its empirical covariance `Σ`) is computed in an
unconstrained transformed space so proposals automatically satisfy the
constraints; `log_pratio` then carries the resulting Jacobian correction. The
target log-density is unaffected either way.
"""
function autoRW(state::SMCState, targets, min_step=1e-3, bounds=nothing)
    store = state.store
    N = nparticles(store)
    min_step = _scalar_arg(min_step)
    d = length(targets)

    if bounds === nothing
        z_old = _column_matrix(store, targets)
        changes = _adaptive_changes(state, z_old, min_step, d)
        proposed = Dict{Symbol,Vector{Float64}}()
        for (i, c) in enumerate(targets)
            proposed[c] = @view(z_old[:, i]) .+ @view(changes[i, :])
        end
        return proposed, zeros(N)
    end

    bnds = _normalize_bounds(bounds, d)
    z_old = _unconstrained_matrix(store, targets, bnds)
    changes = _adaptive_changes(state, z_old, min_step, d)

    return _finish_proposal(z_old, changes, targets, bnds)
end

"""
    default_proposals

`NamedTuple` of the built-in `Move` proposal kernels (`RW`, `autoRW`), used as
the fallback table `@model`-generated functions merge a user-supplied
`proposals` `NamedTuple` into (user entries override same-named defaults).
"""
default_proposals = (
    RW=RW,
    autoRW=autoRW,
)
