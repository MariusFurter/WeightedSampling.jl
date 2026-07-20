"""
Particle-analysis utilities operating on an `SMCState`: weighted expectations,
sampling, and descriptive statistics over the particles.
"""

"""
    expectation(values, weights)

Weighted expectation of `values` with respect to (unnormalized) log-`weights`.
"""
expectation(values, weights) = sum(values .* exp_norm(weights))

"""
    log_evidence(state)

Log-evidence (log marginal likelihood) estimate accumulated in `state`:
`logsumexp(state.weights) - log(N)`, where `N` is the number of particles.
This quantity is preserved across `Resample` steps, so it can be called at
any point during or after a run.
"""
log_evidence(state) = logsumexp(state.weights) - log(length(state.weights))

"""
    @E(f, state)

Compute the weighted expectation of an anonymous function `f` with respect to
the particles held in `state::SMCState`.

# Arguments
- `f`: anonymous function whose argument names reference particle-variable
  column names in `state.store`, e.g. `x -> x^2` or `(x, y) -> x + y`.
- `state`: an `SMCState`.

# Returns
Weighted expectation ``\\mathbb{E}[f] = \\sum_{i=1}^N w_i f(x_i)``, where
``w_i`` are normalized particle weights (`exp_norm(state.weights)`).

# Examples
```julia
posterior_mean = @E(x -> x, state)
second_moment = @E(x -> x^2, state)
cross_term = @E((x, y) -> x * y, state)
```
"""
macro E(f, state)
    if @capture(f, arg_Symbol -> body_)
        args = [arg]
    else
        @capture(f, (args__,) -> body_) ||
            error("@E requires a function of the form (args...) -> expr")
    end

    body = MacroTools.striplines(body)
    if body isa Expr && body.head == :block && length(body.args) == 1
        body = body.args[1]
    end

    particle_vars = Set{Symbol}(args)
    body_vectorized = vectorize(body, particle_vars, Set{Symbol}())

    return esc(quote
        let state = $state
            weights = WeightedSampling.exp_norm(state.weights)
            values = $body_vectorized
            sum(values .* weights)
        end
    end)
end

"""
    DataFrame(state::SMCState) -> DataFrame

Export all particles as a `DataFrame`, one column per entry of
`colnames(state.store)`, plus a `:log_weight` column holding the raw
(unnormalized) log-weights `state.weights`. No resampling/normalization is
applied — use [`sample`](@ref) to draw a weighted sample instead.

# Examples
```julia
df = DataFrame(state)
```
"""
function DataFrames.DataFrame(state::SMCState)
    store = state.store
    df = DataFrame(Dict(c => getcol(store, c) for c in colnames(store)))
    df.log_weight = state.weights
    return df
end

"""
    sample(state::SMCState, n::Int; replace::Bool=true)

Randomly sample `n` particles from `state` according to their (normalized)
weights, returning a `DataFrame` with one column per entry of
`colnames(state.store)`.

# Examples
```julia
posterior_samples = sample(state, 1000)
```
"""
function StatsBase.sample(state::SMCState, n::Int; replace::Bool=true)
    store = state.store
    N = nparticles(store)

    if n <= 0
        throw(ArgumentError("Number of samples must be positive"))
    end

    if !replace && n > N
        throw(ArgumentError("Cannot sample $n particles without replacement from $N particles"))
    end

    normalized_weights = exp_norm(state.weights)
    indices = StatsBase.sample(1:N, StatsBase.Weights(normalized_weights), n; replace=replace)

    return DataFrame(Dict(c => getcol(store, c)[indices] for c in colnames(store)))
end

const _SPARK_CHARS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

"""
    _sparkline(counts::AbstractVector{<:Real}) -> String

Render nonnegative `counts` as a compact one-line bar chart using Unicode
block characters, scaled so the largest count maps to a full-height block.
"""
function _sparkline(counts::AbstractVector{<:Real})
    maxc = maximum(counts)
    maxc <= 0 && return join(fill(_SPARK_CHARS[1], length(counts)))
    levels = clamp.(ceil.(Int, counts ./ maxc .* length(_SPARK_CHARS)), 1, length(_SPARK_CHARS))
    return join(_SPARK_CHARS[levels])
end

"""
    _histogram(values, weights; nbins=8) -> String

Weighted sparkline histogram of `values` binned into `nbins` equal-width bins
spanning `extrema(values)`.
"""
function _histogram(values, weights; nbins::Int=8)
    lo, hi = extrema(values)
    lo == hi && return _sparkline(fill(sum(weights), nbins))

    edges = range(lo, hi, length=nbins + 1)
    counts = zeros(nbins)
    for (v, w) in zip(values, weights)
        bin = clamp(searchsortedlast(edges, v), 1, nbins)
        counts[bin] += w
    end
    return _sparkline(counts)
end

"""
    describe(state::SMCState; cols=nothing)

Compute weighted descriptive statistics for particle variables in `state`.

# Arguments
- `state::SMCState`.
- `cols=nothing`: Vector of column names to analyze (default: all numeric
  columns in `state.store`).

# Returns
DataFrame with columns:
- `variable::Symbol`: Variable name
- `mean`: Weighted mean (scalar or vector)
- `median`: Weighted median (scalar or vector)
- `std`: Weighted standard deviation (scalar or vector)
- `min`: Minimum value (scalar or vector)
- `max`: Maximum value (scalar or vector)
- `hist::String`: Compact weighted sparkline histogram (8 equal-width bins;
  scalar variables only, empty for vector-valued variables).
- `ess::Float64`: Effective sample size ``\\text{ESS} = N \\cdot \\text{ess\\_perc}``

# Examples
```julia
describe(state)
```

Handles both scalar and vector-valued particle variables; vector variables
have statistics computed component-wise.
"""
function DataFrames.describe(state::SMCState; cols=nothing)
    store = state.store
    N = nparticles(store)

    if N == 0
        throw(ArgumentError("Store cannot be empty"))
    end

    normalized_weights = exp_norm(state.weights)
    weights_obj = StatsBase.Weights(normalized_weights)

    function is_numeric_column(name)
        col_type = eltype(getcol(store, name))
        col_type <: Real && return true
        if col_type <: AbstractArray
            return eltype(col_type) <: Real
        end
        return false
    end

    if cols === nothing
        cols = [c for c in colnames(store) if is_numeric_column(c)]
    else
        for col in cols
            hascol(store, col) || throw(ArgumentError("Column $col not found in store"))
            is_numeric_column(col) || throw(ArgumentError("Column $col is not numeric or does not contain numeric elements"))
        end
    end

    if isempty(cols)
        @warn "No numeric columns found to analyze"
        return DataFrame(variable=Symbol[], mean=Any[], median=Any[],
            std=Any[], min=Any[], max=Any[], hist=String[], ess=Float64[])
    end

    ess_val = N * ess_perc(normalized_weights)

    variables = Symbol[]
    means = Any[]
    medians = Any[]
    stds = Any[]
    mins = Any[]
    maxs = Any[]
    hists = String[]
    ess_vals = Float64[]

    for col in cols
        values = getcol(store, col)
        col_type = eltype(values)

        if col_type <: Real
            push!(variables, col)
            push!(means, StatsBase.mean(values, weights_obj))
            push!(medians, StatsBase.quantile(values, weights_obj, 0.5))
            push!(stds, StatsBase.std(values, weights_obj, corrected=false))
            push!(mins, minimum(values))
            push!(maxs, maximum(values))
            push!(hists, _histogram(values, normalized_weights))
            push!(ess_vals, ess_val)

        elseif col_type <: AbstractArray && eltype(col_type) <: Real
            if all(v -> length(v) == length(values[1]), values)
                value_matrix = hcat([v for v in values]...)'
                n_dims = size(value_matrix, 2)

                component_means = Vector{Float64}(undef, n_dims)
                component_medians = Vector{Float64}(undef, n_dims)
                component_stds = Vector{Float64}(undef, n_dims)
                component_mins = Vector{Float64}(undef, n_dims)
                component_maxs = Vector{Float64}(undef, n_dims)

                for dim in 1:n_dims
                    dim_values = value_matrix[:, dim]
                    component_means[dim] = StatsBase.mean(dim_values, weights_obj)
                    component_medians[dim] = StatsBase.quantile(dim_values, weights_obj, 0.5)
                    component_stds[dim] = StatsBase.std(dim_values, weights_obj, corrected=false)
                    component_mins[dim] = minimum(dim_values)
                    component_maxs[dim] = maximum(dim_values)
                end

                push!(variables, col)
                push!(means, component_means)
                push!(medians, component_medians)
                push!(stds, component_stds)
                push!(mins, component_mins)
                push!(maxs, component_maxs)
                push!(hists, "")
                push!(ess_vals, ess_val)
            else
                @warn "Skipping column $col: vectors have inconsistent lengths"
                continue
            end
        end
    end

    return DataFrame(
        variable=variables,
        mean=means,
        median=medians,
        std=stds,
        min=mins,
        max=maxs,
        hist=hists,
        ess=ess_vals
    )
end
