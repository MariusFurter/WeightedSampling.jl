"""
`@model` — macro-based construction of `ParticleTransformer` trees from a
concise probabilistic-program DSL. `@model function name(...) ... end` defines
a function that, when called, builds (but does not run) a `Sequence` of
transformers for the model.

# Supported statements
- `x .= expr` — deterministic broadcast assignment (`Assign`).
- `x ~ f(args...)` — sample from kernel `f`, updating weights if `f` weights
  (`Sample`).
- `_ ~ f(args...)` — weight without sampling (`Weight`).
- `expr => f(args...)` — condition on an observation (`Observe`).
- `x << q(args...)` / `(x, y) << q(args...)` — Metropolis–Hastings move with
  proposal `q` (`Move`). Targets must already be particle variables or
  registered dynamic families `x{e}`; a value-level accessor target
  (`x[e]`/`x.p`) is an error, since a move rewrites a whole column. An
  optional reserved `diversity=threshold` keyword on the proposal call (e.g.
  `x << autoRW(; diversity=0.9)` or `x << RW(0.1; diversity=0.9)`) self-gates
  the move on particle diversity — see "Diversity-gated moves" below. Without
  it, the move always runs (previous behavior).
- `x = expr` (and compound `x += e`, …) — build-time locals; a
  particle-variable target is an error (use `.=` instead), and referencing a
  particle variable or dynamic-variable family `x{e}` anywhere on the
  right-hand side is also an error (use `.=`/`~` instead), since a plain `=`
  only runs once at build time and cannot read a per-particle column.
- `for x in coll ... end` — a `Loop` (not unrolled).
- `if cond ... end` — a `Cond` (no `else`). `cond` must not reference a
  particle variable, but may reference `resampled` (e.g. `if resampled ...`).
- `Resample()` — passed through verbatim.

# Value-level accessors and dynamic variables
Accessors `[]`/`.` are supported on reads and writes to any depth of chaining
(`x[i].p`, `x.p[i]`, `x{i}[j]`, …). Writes mutate the existing element in
place, so the target column must already exist (property writes require
reference-type elements, e.g. mutable structs).

Dynamic-variable families `x{e}` create a column per index (`x{7}` resolves to
`:x_7`). `e` must be a build-time expression that does not depend on a particle
variable, and a base symbol may not be used both as a plain particle variable
and as a dynamic family.

# Resampling
A `Resample()` step is auto-inserted after every weighting statement (`~`,
`=>`); each is ESS-gated, so it only reshuffles when particles degenerate. No
`Resample` is auto-inserted around a `Move` (`<<`).

# Diversity-gated moves
A `diversity=threshold` keyword on a move's proposal call (see `<<` above)
makes the move self-gating: `Move.apply!` skips the whole step whenever
every target column already has a fraction of unique values `>= threshold`
(see [`marginal_diversity`](@ref); for a joint move over several targets, the
MINIMUM per-target diversity is used, not the diversity of the joint tuples,
since the latter can look high even when every individual target has
collapsed). This makes the classic `if resampled; x << q(); end` wrapper
unnecessary — `x << q(...; diversity=0.9)` can be written directly at the top
level of the loop body and will only actually run the (comparatively
expensive) proposal+accept/reject step when particle collapse has actually
reduced diversity below the threshold, rather than on every resample event
regardless of severity. `if resampled` still works as before (and is still
required if you want unconditional/ungated moves gated only by whether a
resample just happened).

# Kernel and proposal resolution
The generated function takes `kernels::NamedTuple` and `proposals::NamedTuple`
keyword arguments. When `f` in `x ~ f(...)` (or `q` in `x << q(...)`) is a bare
symbol present in the table, that entry is used; otherwise the symbol is
evaluated directly (so `x ~ Normal(...)` works via the default kernels, or with
no table at all for a locally-defined kernel). Resolution happens once, when the
model function runs.

# Not supported
Dotted compound assignments (`x .+= …`) raise a clear error; write `x .= x .+ …`.
"""

using MacroTools: MacroTools, @capture

# ---------------------------------------------------------------------------
# `vectorize` turns a raw expression into an expression (evaluated inside a
# `state -> ...` closure) that computes the promoted (vectorized) value.
# ---------------------------------------------------------------------------

"""
    dynname(base::Symbol, idx) -> Symbol

The single source of truth for constructing a dynamic-variable family's
concrete column name (`x{7}` -> `:x_7`). Every getter/setter/`score!` path
goes through this function so they cannot drift out of sync. Underscore-
separated, so `x{1}` followed by `0` (i.e. `x_1` vs `x_10`) and a plain column
literally named `x1` can never be confused with a dynamic family member.
"""
dynname(base::Symbol, idx) = Symbol(base, :_, idx)

"""
    contains_particle_var(expr, particle_vars::Set{Symbol}) -> Bool

Whether `expr` references any symbol currently tracked as a particle variable.
`particle_vars` is an inclusion set that grows as `.=`/`~` targets are
encountered during the single top-to-bottom parse. A dynamic-variable access
`x{e}` (`:curly` head) always counts as containing a particle variable — it is
always a column access, regardless of whether `x` itself is in `particle_vars`
(dynamic families are tracked separately).
"""
function contains_particle_var(expr, particle_vars::Set{Symbol})
    if expr isa Symbol
        return expr in particle_vars
    elseif expr isa Expr
        expr.head == :curly && return true
        return any(a -> contains_particle_var(a, particle_vars), expr.args)
    else
        return false
    end
end

"""
    vectorize(expr, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol}) -> Expr

Build an expression (referencing a free variable `state`) computing the
promoted value of `expr`:

- no particle variable anywhere in `expr` → `Ref(expr)` (computed once,
  broadcasts as one value per particle; also handles array-valued locals,
  which are wrapped whole, never element-broadcast).
- a bare particle-variable symbol → `getcol(state.store, :sym)`.
- `f(args...)` with ≥1 particle variable among `args` → recursively vectorize
  each argument and broadcast: `f.(vectorize(arg1), vectorize(arg2), ...)`.
- `b[e]` → `getindex.(vectorize(b), vectorize(e))`: `b` and `e` are each
  recursively vectorized, so the base need not be a bare symbol (`x.p[e]`,
  `x[i][j]`, …) and a particle-dependent index broadcasts correctly.
- `b.p` → `getproperty.(vectorize(b), :p)`, same recursive base handling.
- `x{e}` (`:curly`) → `getcol(state.store, dynname(:x, e))`: `e` must be a
  build-time-evaluable expression not depending on a particle variable, and
  `x` must already be a registered dynamic-variable family.
- `[a, b, ...]` (`vect`) containing a particle variable → elementwise combine
  the vectorized args into an array per particle.

Anything else containing a particle variable (e.g. `(a, b)` tuples) raises an
error.
"""
function vectorize(expr, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol})
    if !contains_particle_var(expr, particle_vars)
        return :(Ref($expr))
    end

    if expr isa Symbol
        return :(WeightedSampling.getcol(state.store, $(QuoteNode(expr))))
    elseif expr isa Expr
        if expr.head == :call
            f = expr.args[1]
            vec_args = [vectorize(a, particle_vars, dynamic_families) for a in expr.args[2:end]]
            # Emit a fused dot-call `f.(vec_args...)` (AST: `Expr(:., f, tuple)`)
            # rather than an eager `broadcast(f, vec_args...)`. Nested vectorized
            # calls then fuse into a SINGLE broadcast loop, so intermediate
            # subexpressions (e.g. `β * x` inside `α + β * x`) are never
            # materialized — roughly halving broadcast-argument allocation in
            # both the forward `apply!` pass and the `score!` walk. At least one
            # argument contains a particle variable (else the top-of-function
            # `Ref(expr)` guard already fired), so the broadcast has an
            # array-shaped leaf and produces `N` results, never collapsing to a
            # single scalar the way an all-`Ref` fused call would.
            return Expr(:., f, Expr(:tuple, vec_args...))
        elseif expr.head == :ref
            length(expr.args) == 2 ||
                error("Unsupported indexing expression (only a single index `x[e]` is supported): $expr")
            base, idx = expr.args[1], expr.args[2]
            vec_base = vectorize(base, particle_vars, dynamic_families)
            vec_idx = vectorize(idx, particle_vars, dynamic_families)
            return :(getindex.($vec_base, $vec_idx))
        elseif expr.head == :.
            base, prop = expr.args[1], expr.args[2]
            vec_base = vectorize(base, particle_vars, dynamic_families)
            return :(getproperty.($vec_base, $prop))
        elseif expr.head == :curly
            length(expr.args) == 2 && expr.args[1] isa Symbol ||
                error("Unsupported dynamic-variable expression (expected `x{e}`): $expr")
            base, idx = expr.args[1], expr.args[2]
            contains_particle_var(idx, particle_vars) &&
                error("Dynamic-variable index in `$expr` must not depend on a particle variable " *
                      "(a column name cannot vary per particle): $expr")
            base in dynamic_families ||
                error("`$base` is not a registered dynamic-variable family; assign `$base" * "{...} .= ...` " *
                      "(or `~`) first before reading `$expr`")
            return :(WeightedSampling.getcol(state.store, WeightedSampling.dynname($(QuoteNode(base)), $idx)))
        elseif expr.head == :vect
            vec_args = [vectorize(a, particle_vars, dynamic_families) for a in expr.args]
            return :(broadcast((xs...) -> [xs...], $(vec_args...)))
        else
            error("Unsupported expression containing a particle variable: $expr")
        end
    else
        return :(Ref($expr))
    end
end

# ---------------------------------------------------------------------------
# Dynamic-variable LHS targets (`x{e} .= rhs`, `x{e} ~ f(args...)`). `x{e}`
# always resolves to a plain `Symbol` (`dynname(:x, e)`), so writes are
# ordinary `Assign`/`Sample` targets; see `dynname`/`vectorize`'s `:curly`
# case for reads.
# ---------------------------------------------------------------------------

"""
    dynamic_curly_target(lhs::Expr, particle_vars, dynamic_families, stmt) -> (base::Symbol, idx)

Validate a dynamic-variable assignment/sampling target `x{e}` and return
`(base, idx)`. The index `idx` must not depend on a particle variable (a
column name cannot vary per particle), and `base` must not already be a plain
particle variable (a base symbol may not be used both as a plain column and a
dynamic family). Does not itself register `base` in `dynamic_families`;
callers do that after this check succeeds.
"""
function dynamic_curly_target(lhs::Expr, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol}, stmt)
    length(lhs.args) == 2 && lhs.args[1] isa Symbol ||
        error("Unsupported dynamic-variable target (expected `x{e}`): $lhs")
    base, idx = lhs.args[1], lhs.args[2]
    contains_particle_var(idx, particle_vars) &&
        error("Dynamic-variable index in `$lhs` must not depend on a particle variable " *
              "(a column name cannot vary per particle): $stmt")
    base in particle_vars &&
        error("`$base` is already a plain particle variable; it cannot also be used as a dynamic-variable " *
              "family `$base" * "{...}`: $stmt")
    return base, idx
end

# ---------------------------------------------------------------------------
# Value-accessor LHS writes (`x[e] .= rhs`, `x.p .= rhs`, and chains thereof).
# Unlike a plain `x .= rhs`, these mutate elements already stored in an
# existing column (`setindex!`/`setproperty!`) rather than replacing the
# column, so the target column must already exist.
# ---------------------------------------------------------------------------

"""
    accessor_root(expr, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol}, stmt) -> Nothing

Validate that the innermost base of a chained value-accessor expression
(`x`, `x{e}`, `x[e]`, `x.p`, `x{e}[j]`, `x.p[e]`, …) refers to an
already-existing column: either a plain particle variable or a registered
dynamic-variable family member `x{e}`. Errors otherwise.
"""
function accessor_root(expr, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol}, stmt)
    if expr isa Symbol
        expr in particle_vars ||
            error("Accessor assignment target requires `$expr` to already be a particle variable " *
                  "(assign it first via `.=`/`~`): $stmt")
        return nothing
    elseif expr isa Expr && expr.head in (:ref, :.)
        return accessor_root(expr.args[1], particle_vars, dynamic_families, stmt)
    elseif expr isa Expr && expr.head == :curly
        base, _ = dynamic_curly_target(expr, particle_vars, dynamic_families, stmt)
        base in dynamic_families ||
            error("`$expr` is not a registered dynamic-variable family; assign `$base" * "{...} .= ...`/`~` " *
                  "first before using it in an accessor: $stmt")
        return nothing
    else
        error("Unsupported accessor assignment target (must resolve to a plain particle variable or a " *
              "dynamic-variable family `x{e}`): $expr")
    end
end

"""
    accessor_write_fn(lhs, rhs, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol}) -> Expr

Build the `state -> nothing` mutation closure for an accessor-write statement
`lhs .= rhs`, where `lhs` is `x[e]` or `x.p`. The *container* (everything but
the outermost accessor) is vectorized with the ordinary `vectorize` recursion,
so chains (`x[e].p .= rhs`, `x.p[e] .= rhs`) resolve automatically: the inner
accessor is read (producing an array of the existing elements) and the outer
one performs the in-place mutation.
"""
function accessor_write_fn(lhs::Expr, rhs, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol})
    rhs_expr = vectorize(rhs, particle_vars, dynamic_families)
    if lhs.head == :ref
        length(lhs.args) == 2 ||
            error("Unsupported indexed assignment target (only a single index `x[e] .= ...` is supported): $lhs")
        base, idx = lhs.args[1], lhs.args[2]
        container_expr = vectorize(base, particle_vars, dynamic_families)
        idx_expr = vectorize(idx, particle_vars, dynamic_families)
        return :(state -> (setindex!.($container_expr, $rhs_expr, $idx_expr); nothing))
    elseif lhs.head == :.
        base, prop = lhs.args[1], lhs.args[2]
        container_expr = vectorize(base, particle_vars, dynamic_families)
        return :(state -> (setproperty!.($container_expr, $prop, $rhs_expr); nothing))
    else
        error("Unsupported accessor assignment target: $lhs")
    end
end

"""
    accessor_sample_fns(lhs::Expr, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol}) -> (readfn::Expr, writefn::Expr)

Build the `state -> values` read closure and `(state, values) -> nothing`
write closure for an accessor-target sampling statement `lhs ~ f(args...)`,
where `lhs` is `x[e]` or `x.p`, for use by `AccessorSample`. The *container*
(everything but the outermost accessor) is vectorized with the ordinary
`vectorize` recursion (exactly as in `accessor_write_fn`), so chains resolve
automatically (`x[e].p ~ f(...)`, `x.p[e] ~ f(...)`). `readfn` re-reads the
container fresh on every call (not cached), matching `writefn`, since the
underlying store can be reordered by resampling between statements.
"""
function accessor_sample_fns(lhs::Expr, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol})
    if lhs.head == :ref
        length(lhs.args) == 2 ||
            error("Unsupported indexed sampling target (only a single index `x[e] ~ ...` is supported): $lhs")
        base, idx = lhs.args[1], lhs.args[2]
        container_expr = vectorize(base, particle_vars, dynamic_families)
        idx_expr = vectorize(idx, particle_vars, dynamic_families)
        readfn = :(state -> getindex.($container_expr, $idx_expr))
        writefn = :((state, values) -> (setindex!.($container_expr, values, $idx_expr); nothing))
        return readfn, writefn
    elseif lhs.head == :.
        base, prop = lhs.args[1], lhs.args[2]
        container_expr = vectorize(base, particle_vars, dynamic_families)
        readfn = :(state -> getproperty.($container_expr, $prop))
        writefn = :((state, values) -> (setproperty!.($container_expr, $prop, values); nothing))
        return readfn, writefn
    else
        error("Unsupported accessor sampling target: $lhs")
    end
end

# ---------------------------------------------------------------------------
# `if` condition promotion: only the special symbol `resampled` is rewritten
# (to `state.resampled`); everything else is left as-is (a build-time-captured
# local). `if` conditions must not reference particle variables — checked
# separately in `walk_body`.
# ---------------------------------------------------------------------------

"""
    replace_state_symbols(expr)

Replace bare occurrences of the symbol `resampled` with `state.resampled` in
`expr` (an `if` condition), leaving everything else untouched.
"""
function replace_state_symbols(expr)
    if expr isa Symbol
        return expr === :resampled ? :(state.resampled) : expr
    elseif expr isa Expr
        return Expr(expr.head, map(replace_state_symbols, expr.args)...)
    else
        return expr
    end
end

# ---------------------------------------------------------------------------
# Kernel resolution
# ---------------------------------------------------------------------------

"""
    kernel_expr(f, kernels_var::Symbol) -> Expr

Build an expression resolving `f` to a `WeightedKernel`, evaluated once when
the model function runs. If `f` is a bare `Symbol`, prefer a same-named
entry in the `kernels` `NamedTuple` argument, falling back to `f` itself as
an expression (so plain kernel variables/constants work with no table at
all).
"""
function kernel_expr(f, kernels_var::Symbol)
    if f isa Symbol
        return :($(Base.hasproperty)($kernels_var, $(QuoteNode(f))) ? getproperty($kernels_var, $(QuoteNode(f))) : $(f))
    else
        return f
    end
end

"""
    proposal_expr(q, proposals_var::Symbol) -> Expr

Build an expression resolving `q` to a proposal function, evaluated once when
the model function runs. Mirrors `kernel_expr`: if `q` is a bare `Symbol`,
prefer a same-named entry in the `proposals` `NamedTuple` argument, falling
back to `q` itself as an expression (so `x << RW(0.1)` / `x << autoRW()` work
with no table at all).
"""
function proposal_expr(q, proposals_var::Symbol)
    if q isa Symbol
        return :($(Base.hasproperty)($proposals_var, $(QuoteNode(q))) ? getproperty($proposals_var, $(QuoteNode(q))) : $(q))
    else
        return q
    end
end

"""
    split_move_call_args(args) -> (pos_args::Vector, diversity_expr)

Split a captured `<< q(args__)` argument list into the plain positional
proposal arguments and a reserved `diversity=...` keyword argument, if
present. Handles BOTH Julia keyword-argument syntaxes, which parse
differently: `autoRW(1e-3; diversity=0.9)` (semicolon) captures the keyword(s)
as a single leading `Expr(:parameters, ...)` element of `args`, while
`autoRW(1e-3, diversity=0.9)` (comma) captures each keyword as its own
`Expr(:kw, ...)` element interleaved among the positional args (not
necessarily at the front) — confirmed via `dump` probing, not assumed. Both
forms are pulled out here so `diversity=` can be written either way.
Returns `diversity_expr = nothing` (a literal, not `:(nothing)`) when no
`diversity=` keyword is given, so the generated `Move` gets `nothing` as its
`diversity_threshold` (always-move, backward-compatible default).
"""
function split_move_call_args(args)
    pos_args = []
    kw_exprs = []
    for a in args
        if a isa Expr && a.head == :parameters
            append!(kw_exprs, a.args)
        elseif a isa Expr && a.head == :kw
            push!(kw_exprs, a)
        else
            push!(pos_args, a)
        end
    end

    diversity_expr = nothing
    for p in kw_exprs
        @capture(p, name_ = value_) || error("Unsupported keyword argument in move call: $p")
        name === :diversity ||
            error("Unsupported keyword argument `$name` in move call (only `diversity=...` is supported)")
        diversity_expr = value
    end

    return pos_args, diversity_expr
end

# ---------------------------------------------------------------------------
# Move (`<<`) target resolution: a target is either a plain particle variable
# or a dynamic-variable family member `x{e}` — never a value-level accessor
# (`x[e]`/`x.p`), since a `Move` proposes/writes a whole column via
# `broadcast_setcol!`, not an in-place element mutation.
# ---------------------------------------------------------------------------

"""
    move_target_expr(t, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol}, stmt) -> Expr

Validate one `Move` target `t` (an element of `x << q(...)` or
`(x, y, ...) << q(...)`) and return an expression evaluating to its concrete
column `Symbol`:

- a plain `Symbol` → must already be `in particle_vars`, returns
  `QuoteNode(t)`.
- `x{e}` (`:curly`) → must have an index that doesn't depend on a particle
  variable and be an already-registered dynamic family, returns
  `dynname(:base, e)`.
- anything else (in particular `x[e]`/`x.p`) is an error: a `Move` writes a
  whole column, not an element.
"""
function move_target_expr(t, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol}, stmt)
    if t isa Symbol
        t in particle_vars ||
            error("Move target `$t` is not a particle variable (must already be assigned via `.=`/`~` before `<<`): $stmt")
        return QuoteNode(t)
    elseif t isa Expr && t.head == :curly
        base, idx = dynamic_curly_target(t, particle_vars, dynamic_families, stmt)
        base in dynamic_families ||
            error("Move target `$t` is not a registered dynamic-variable family (must already be assigned via " *
                  "`$base" * "{...} .= ...`/`~` before `<<`): $stmt")
        return :(WeightedSampling.dynname($(QuoteNode(base)), $idx))
    else
        error("Unsupported move target `$t` (only a plain variable `x` or a dynamic-variable family `x{e}` " *
              "are supported — NOT a value-level accessor `x[e]`/`x.p`, since a `Move` writes a whole column): $stmt")
    end
end

# ---------------------------------------------------------------------------
# Per-statement transformer generation
# ---------------------------------------------------------------------------

"""
    gen_step(stmt, particle_vars, dynamic_families, kernels_var) -> Expr

Turn one particle-affecting statement (`.=`, `~`, `_ ~`, `=>`) into an
expression constructing the corresponding `ParticleTransformer`. Mutates
`particle_vars`, adding any new plain `.=`/`~` target, or `dynamic_families`,
adding any new dynamic-variable family base (`x{e}` target) — a base symbol
may never be registered in both sets.
"""
function gen_step(stmt, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol}, kernels_var::Symbol)
    if @capture(stmt, lhsexpr_ => f_(args__))
        kexpr = kernel_expr(f, kernels_var)
        lhsfn = :(state -> $(vectorize(lhsexpr, particle_vars, dynamic_families)))
        argfn = :(state -> ($(map(a -> vectorize(a, particle_vars, dynamic_families), args)...),))
        return :(WeightedSampling.Observe($lhsfn, $kexpr, $argfn))

    elseif @capture(stmt, lhs_ ~ f_(args__))
        kexpr = kernel_expr(f, kernels_var)
        if lhs == :_
            argfn = :(state -> ($(map(a -> vectorize(a, particle_vars, dynamic_families), args)...),))
            return :(WeightedSampling.Weight($kexpr, $argfn))
        elseif lhs isa Symbol
            lhs in dynamic_families &&
                error("`$lhs` is already a dynamic-variable family (`$lhs" * "{...}`); it cannot also be " *
                      "used as a plain particle variable: $stmt")
            push!(particle_vars, lhs)
            argfn = :(state -> ($(map(a -> vectorize(a, particle_vars, dynamic_families), args)...),))
            return :(WeightedSampling.Sample($(QuoteNode(lhs)), $kexpr, $argfn))
        elseif lhs isa Expr && lhs.head == :curly
            base, idx = dynamic_curly_target(lhs, particle_vars, dynamic_families, stmt)
            push!(dynamic_families, base)
            argfn = :(state -> ($(map(a -> vectorize(a, particle_vars, dynamic_families), args)...),))
            return :(WeightedSampling.Sample(WeightedSampling.dynname($(QuoteNode(base)), $idx), $kexpr, $argfn))
        elseif lhs isa Expr && lhs.head in (:ref, :.)
            accessor_root(lhs, particle_vars, dynamic_families, stmt)
            readfn, writefn = accessor_sample_fns(lhs, particle_vars, dynamic_families)
            argfn = :(state -> ($(map(a -> vectorize(a, particle_vars, dynamic_families), args)...),))
            return :(WeightedSampling.AccessorSample($readfn, $writefn, $kexpr, $argfn))
        else
            error("Unsupported sampling target (only plain variables `x`, dynamic families `x{e}`, `x[e]`, " *
                  "`x.p`, or chains thereof are supported): $lhs")
        end

    elseif @capture(stmt, lhs_ .= rhs_)
        if lhs isa Symbol
            lhs in dynamic_families &&
                error("`$lhs` is already a dynamic-variable family (`$lhs" * "{...}`); it cannot also be " *
                      "used as a plain particle variable: $stmt")
            argfn = :(state -> $(vectorize(rhs, particle_vars, dynamic_families)))
            push!(particle_vars, lhs)
            return :(WeightedSampling.Assign($(QuoteNode(lhs)), $argfn))
        elseif lhs isa Expr && lhs.head == :curly
            base, idx = dynamic_curly_target(lhs, particle_vars, dynamic_families, stmt)
            push!(dynamic_families, base)
            argfn = :(state -> $(vectorize(rhs, particle_vars, dynamic_families)))
            return :(WeightedSampling.Assign(WeightedSampling.dynname($(QuoteNode(base)), $idx), $argfn))
        elseif lhs isa Expr && lhs.head in (:ref, :.)
            accessor_root(lhs, particle_vars, dynamic_families, stmt)
            return :(WeightedSampling.AccessorAssign($(accessor_write_fn(lhs, rhs, particle_vars, dynamic_families))))
        else
            error("Unsupported assignment target (only plain variables `x`, dynamic families `x{e}`, `x[e]`, " *
                  "`x.p`, or chains thereof are supported for now): $lhs")
        end

    else
        error("Unsupported statement in @model body: $stmt")
    end
end

is_particle_stmt(stmt) =
    @capture(stmt, _ => _(__)) || @capture(stmt, _ ~ _(__)) || @capture(stmt, _ .= _)

"""
    is_weighting_stmt(stmt) -> Bool

Whether `stmt` is a statement that can change particle weights — an
observation `expr => f(args)` or a sampling/weighting step `x ~ f(args)` /
`_ ~ f(args)`. Used to decide where `@model` auto-inserts a
`Resample()` step. Deterministic assignments (`x .= …`) are excluded (they
never touch the weights).
"""
is_weighting_stmt(stmt) =
    @capture(stmt, _ => _(__)) || @capture(stmt, _ ~ _(__))


# Non-dotted compound updating-assignment heads (`x += e`, `x *= e`, …) and
# their dotted counterparts (`x .+= e`, …). Non-dotted forms are treated as
# build-time locals (guarded against particle-variable targets — see
# `walk_body`); dotted forms target particle columns and are NOT supported yet
# (a clear error is raised, suggesting `x .= x .+ e`).
const _UPDATE_OPS = (:+, :-, :*, :/, :\, :^, :%, :÷, :&, :|, :⊻, :>>, :<<, :>>>)
const UPDATE_ASSIGN_HEADS = Tuple(Symbol(string(op) * "=") for op in _UPDATE_OPS)
const DOTTED_UPDATE_HEADS = Tuple(Symbol("." * string(op) * "=") for op in _UPDATE_OPS)

is_update_assign(stmt) = stmt isa Expr && stmt.head in UPDATE_ASSIGN_HEADS
is_dotted_update(stmt) = stmt isa Expr && stmt.head in DOTTED_UPDATE_HEADS

"""
    collect_particle_vars(expr, particle_vars, dynamic_families=Set{Symbol}()) -> Set{Symbol}

The subset of `particle_vars` that `expr` references, plus the base symbol of
any dynamic-variable family read (`x{e}` where `x in dynamic_families`) found
in `expr`. Used only for building clear error messages when a plain/compound
assignment targets (LHS) or reads (RHS) a particle-backed value — see the
`= `/compound-update branch of `walk_body`. `dynamic_families` defaults to
empty so LHS-only callers need not pass it.
"""
function collect_particle_vars(expr, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol}=Set{Symbol}(),
    acc::Set{Symbol}=Set{Symbol}())
    if expr isa Symbol
        expr in particle_vars && push!(acc, expr)
    elseif expr isa Expr
        if expr.head == :curly && length(expr.args) == 2 && expr.args[1] isa Symbol &&
           expr.args[1] in dynamic_families
            push!(acc, expr.args[1])
        end
        for a in expr.args
            collect_particle_vars(a, particle_vars, dynamic_families, acc)
        end
    end
    return acc
end

# ---------------------------------------------------------------------------
# Body walk: sequences of statements, `for` loops (→ `Loop`, not unrolled)
# ---------------------------------------------------------------------------

"""
    walk_body(body, particle_vars, dynamic_families, kernels_var, proposals_var) -> (stmts::Vector, step_names::Vector{Symbol})

Recursively walk a model body (a `Vector` of statements). Returns:
- `stmts`: build-time Julia statements to execute in order (plain local
  assignments verbatim, transformer steps as `stepname = TransformerExpr(...)`).
- `step_names`: the gensym'd names (in order) of the transformer steps among
  `stmts`, i.e. the ones to collect into the enclosing `Sequence`.

A `Resample()` step is inserted immediately after every weighting statement
(`is_weighting_stmt`), so the resulting `Sequence` resamples after each step
(each `Resample` is ESS-gated, so this only reshuffles when needed). No
`Resample` is auto-inserted around a `Move` (`<<`) step: a move neither
weights nor advances the trace, so it isn't a natural resampling boundary the
way a weighting statement is.

Local (`=`) and compound (`+=`, …) assignments are kept as ordinary Julia
code rather than transformer steps, since they only need to run once (at build
time) to be captured by later closures. A particle-variable target is a
compile-time error (use `.=` instead), and so is a particle variable or
dynamic-variable family reference anywhere on the right-hand side (use
`.=`/`~` instead) — such code would otherwise silently reference an undefined
Julia local (or, worse, an unrelated global) at runtime instead of reading the
actual particle column.
"""
function walk_body(body, particle_vars::Set{Symbol}, dynamic_families::Set{Symbol}, kernels_var::Symbol, proposals_var::Symbol)
    stmts = Expr[]
    step_names = Symbol[]

    for stmt in body
        stmt isa LineNumberNode && continue

        if @capture(stmt, for loopvar_ in coll_
            loopbody__
        end)
            inner_stmts, inner_steps = walk_body(loopbody, particle_vars, dynamic_families, kernels_var, proposals_var)
            # `Loop.bodyfn` is always called with ONE positional argument (the
            # current collection element, `t.bodyfn(x)` in `apply!`/`score!`).
            # A plain loop variable (`for x in coll`) can be used directly as
            # the closure's argument. A destructuring loop variable (`for (x,
            # y) in coll`) needs Julia's single-argument tuple-destructuring
            # syntax `((x, y),) -> ...` — using the bare pattern `(x, y) -> ...`
            # would instead define a 2-ARGUMENT anonymous function, which then
            # fails at the single-argument call site.
            bodyparam = loopvar isa Expr && loopvar.head === :tuple ? Expr(:tuple, loopvar) : loopvar
            bodyfn = :($bodyparam -> begin
                $(inner_stmts...)
                WeightedSampling.Sequence($(inner_steps...))
            end)
            name = gensym(:step)
            push!(stmts, :($name = WeightedSampling.Loop(_state -> $coll, $bodyfn)))
            push!(step_names, name)

        elseif @capture(stmt, if cond_
            ifbody__
        end)
            if contains_particle_var(cond, particle_vars)
                error("`if` condition must not reference a particle variable: $cond")
            end
            inner_stmts, inner_steps = walk_body(ifbody, particle_vars, dynamic_families, kernels_var, proposals_var)
            append!(stmts, inner_stmts)
            predfn = :(state -> $(replace_state_symbols(cond)))
            name = gensym(:step)
            push!(stmts, :($name = WeightedSampling.Cond($predfn, WeightedSampling.Sequence($(inner_steps...)))))
            push!(step_names, name)

        elseif @capture(stmt, lhs_ << q_(args__))
            target_inputs = if lhs isa Symbol || (lhs isa Expr && lhs.head == :curly)
                [lhs]
            elseif lhs isa Expr && lhs.head == :tuple
                lhs.args
            else
                error("Unsupported move target (only `x`, `x{e}`, or `(x, y, ...)` thereof are supported): $stmt")
            end

            target_exprs = [move_target_expr(t, particle_vars, dynamic_families, stmt) for t in target_inputs]

            pos_args, diversity_expr = split_move_call_args(args)
            qexpr = proposal_expr(q, proposals_var)
            targets_expr = Expr(:vect, target_exprs...)
            argfn = :(state -> ($(pos_args...),))
            name = gensym(:step)
            push!(stmts, :($name = WeightedSampling.Move($targets_expr, $qexpr, $argfn, $diversity_expr)))
            push!(step_names, name)

        elseif is_particle_stmt(stmt)
            name = gensym(:step)
            push!(stmts, :($name = $(gen_step(stmt, particle_vars, dynamic_families, kernels_var))))
            push!(step_names, name)
            if is_weighting_stmt(stmt)
                rname = gensym(:resample)
                push!(stmts, :($rname = WeightedSampling.Resample()))
                push!(step_names, rname)
            end

        elseif is_dotted_update(stmt)
            error("Dotted compound assignment (`$(stmt.head)`) is not supported yet; " *
                  "write it out explicitly, e.g. `x .= x .+ e` instead of `x .+= e`: $stmt")

        elseif @capture(stmt, _ = _) || is_update_assign(stmt)
            lhs_hit = collect_particle_vars(stmt.args[1], particle_vars)
            if !isempty(lhs_hit)
                error("Assignment `$stmt` targets particle variable(s) $(collect(lhs_hit)); a plain " *
                      "`=` or a compound update (`+=`, `-=`, …) only binds a build-time local and " *
                      "does NOT update the particle column. Use `x .= …` to update the particle " *
                      "column, or rename the local if you did not mean the particle variable.")
            end
            rhs_hit = collect_particle_vars(stmt.args[2], particle_vars, dynamic_families)
            if !isempty(rhs_hit)
                error("Assignment `$stmt` reads particle variable(s)/dynamic-variable family/families " *
                      "$(collect(rhs_hit)) on its right-hand side; a plain `=` or compound update " *
                      "(`+=`, `-=`, …) only runs once at build time and cannot read a per-particle " *
                      "column. Use `x .= …` or `x ~ …` instead.")
            end
            push!(stmts, stmt)

        elseif stmt isa Expr && stmt.head == :call
            # A plain function call statement (e.g. `Resample()`) is assumed
            # to already construct a `ParticleTransformer` and is passed
            # through as a step verbatim.
            name = gensym(:step)
            if stmt.args[1] === :Resample
                push!(stmts, :($name = WeightedSampling.Resample($(stmt.args[2:end]...))))
            else
                push!(stmts, :($name = $stmt))
            end
            push!(step_names, name)

        else
            error("Unsupported statement in @model body: $stmt")
        end
    end

    return stmts, step_names
end

# ---------------------------------------------------------------------------
# The `@model` macro
# ---------------------------------------------------------------------------

"""
    @model function name(args...; kwargs...)
        # model body
    end

Compiles a model body into a `name(args...; kwargs..., kernels=NamedTuple(), proposals=NamedTuple())`
function that, when called, constructs (but does not run) a `ParticleTransformer`
(a `Sequence`) for that model. Run it with `apply!(name(...), state)`, or, if
the model contains a `Move` (`<<`), `run!(name(...), state)` (needed so
`state.root` is set for `score!`).

A `Resample()` step is auto-inserted after every weighting statement (`~`,
`=>`); each `Resample` is ESS-gated, so this only reshuffles when particles
degenerate. No `Resample` is auto-inserted around a `Move` (`<<`) step.

See the module docstring above (top of this file) for the currently supported
subset of the DSL.
"""
macro model(expr)
    if @capture(expr, function name_(args__; kwargs__)
        body__
    end)
    elseif @capture(expr, function name_(; kwargs__)
        body__
    end)
        args = Symbol[]
    elseif @capture(expr, function name_(args__)
        body__
    end)
        kwargs = Expr[]
    else
        error("Expression must be a function definition")
    end

    kernels_var = :kernels
    proposals_var = :proposals
    particle_vars = Set{Symbol}()
    dynamic_families = Set{Symbol}()
    stmts, step_names = walk_body(body, particle_vars, dynamic_families, kernels_var, proposals_var)

    return esc(quote
        function $name($(args...); $(kwargs...), $kernels_var=NamedTuple(), $proposals_var=NamedTuple())
            $kernels_var = merge(WeightedSampling.default_kernels, $kernels_var)
            $proposals_var = merge(WeightedSampling.default_proposals, $proposals_var)
            $(stmts...)
            WeightedSampling.Sequence($(step_names...))
        end
    end)
end
