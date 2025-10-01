"""
Exponentiate and normalize the log weights in-place.
"""
function exp_norm!(weights::Vector{Float64})
    max_weight = maximum(weights)
    weights .= exp.(weights .- max_weight)
    sum_exp_weights = sum(weights)
    weights ./= sum_exp_weights
    return nothing
end

"""
Exponentiate and normalize the log weights, returning a new vector.
"""
function exp_norm(weights::Vector{Float64})
    max_weight = maximum(weights)
    exp_weights = exp.(weights .- max_weight)
    sum_exp_weights = sum(exp_weights)
    return exp_weights ./ sum_exp_weights
end

expectation(values, weights) = sum(values .* exp_norm(weights))


"""
Compute the weighted expectation of an anonymous function `f` with respect to particle samples. Argument names of `f` reference the corresponding particle values.
"""
macro E(f, particles)
    if @capture(f, arg_Symbol -> body_)
        args = [arg]
    else
        @capture(f, (args__,) -> body_) ||
            error("E macro requires a function of the form (args...) -> expr")
    end

    symbols = (particles=:particles, N=:N)
    body_replaced = WeightedSampling.replace_symbols_in(body, args, symbols)
    return esc(quote
        let particles = $particles
            N = WeightedSampling.nrow(particles)
            weights = WeightedSampling.exp_norm(particles.weights)
            values = $body_replaced
            sum(values .* weights)
        end
    end)
end


"""
Compute the weighted expectation of an expression `f` with respect to particle samples. Variable names in `f` that match column names in `particles` will reference the corresponding particle values.
"""
macro E_old(f, particles)
    f_quoted = QuoteNode(f)

    return esc(quote
        let particles = $particles
            particle_names = Set(Symbol.(names(particles)))
            symbols = (particles=:particles, N=:N)
            f_replaced = WeightedSampling.replace_symbols_in($f_quoted, particle_names, symbols)
            weights = WeightedSampling.exp_norm(particles.weights)

            f_func_code = quote
                (particles) -> $f_replaced
            end
            f_func = eval(f_func_code)

            f_eval = Base.invokelatest(f_func, particles)

            sum(f_eval .* weights)
        end
    end)
end

"""
Compute the weighted expectation of an expression `f` with respect to particle samples. Variable names in `f` not in `exceptions` are interpreted as columns in `particles`.
"""
macro E_except(f, particles, exceptions=Symbol[])
    exceptions = Set(Symbol.(eval(exceptions)))
    symbols = (particles=:particles, N=:N)
    f_replaced = replace_symbols_except(f, exceptions, symbols)
    return esc(
        quote
            let particles = $particles
                weights = WeightedSampling.exp_norm(particles.weights)
                sum($f_replaced .* weights)
            end
        end
    )
end


"""
Randomly sample `n` rows from a DataFrame of weighted particles according to their log weights.
"""
function sample_particles(particles::DataFrame, n::Int; replace::Bool=true)
    if !hasproperty(particles, :weights)
        throw(ArgumentError("DataFrame must have a 'weights' column"))
    end

    if n <= 0
        throw(ArgumentError("Number of samples must be positive"))
    end

    if !replace && n > nrow(particles)
        throw(ArgumentError("Cannot sample $n particles without replacement from $(nrow(particles)) particles"))
    end

    normalized_weights = exp_norm(particles.weights)

    indices = StatsBase.sample(1:nrow(particles), StatsBase.Weights(normalized_weights), n; replace=replace)

    return particles[indices, :]
end


"""
Compute weighted descriptive statistics for particles.
"""
function describe_particles(particles::DataFrame; cols=nothing)
    if !hasproperty(particles, :weights)
        throw(ArgumentError("DataFrame must have a 'weights' column"))
    end

    if nrow(particles) == 0
        throw(ArgumentError("DataFrame cannot be empty"))
    end

    # Get normalized weights
    normalized_weights = exp_norm(particles.weights)
    weights_obj = StatsBase.Weights(normalized_weights)

    # Helper function to check if a column is numeric (scalar or vector)
    function is_numeric_column(col_name)
        col_type = eltype(particles[!, col_name])

        # Check if it's directly a Real type
        if col_type <: Real
            return true
        end

        # Check if it's a vector/array of reals
        if col_type <: AbstractVector
            elem_type = eltype(col_type)
            return elem_type <: Real
        end

        # Check if it's an array of reals (for higher-dimensional arrays)
        if col_type <: AbstractArray
            elem_type = eltype(col_type)
            return elem_type <: Real
        end

        return false
    end

    # Determine columns to analyze
    if cols === nothing
        # Include all numeric columns except :weights
        cols = [Symbol(name) for name in names(particles)
                if name != "weights" && is_numeric_column(name)]
    else
        # Validate provided columns
        for col in cols
            if !hasproperty(particles, col)
                throw(ArgumentError("Column $col not found in DataFrame"))
            end
            if !is_numeric_column(col)
                throw(ArgumentError("Column $col is not numeric or does not contain numeric elements"))
            end
        end
    end

    if isempty(cols)
        @warn "No numeric columns found to analyze"
        return DataFrame(variable=Symbol[], mean=Any[], median=Any[],
            std=Any[], min=Any[], max=Any[], ess=Float64[])
    end

    # Compute effective sample size once
    ess_val = 1.0 / sum(normalized_weights .^ 2)

    # Initialize result vectors
    variables = Symbol[]
    means = Any[]  # Allow both Float64 and Vector{Float64}
    medians = Any[]  # Allow both Float64 and Vector{Float64}
    stds = Any[]  # Allow both Float64 and Vector{Float64}
    mins = Any[]  # Allow both Float64 and Vector{Float64}
    maxs = Any[]  # Allow both Float64 and Vector{Float64}
    ess_vals = Float64[]

    for col in cols
        values = particles[!, col]
        col_type = eltype(values)

        if col_type <: Real
            # Handle scalar values using StatsBase functions

            # Weighted mean
            weighted_mean = StatsBase.mean(values, weights_obj)

            # Weighted median
            weighted_median = StatsBase.quantile(values, weights_obj, 0.5)

            # Weighted standard deviation
            weighted_std = StatsBase.std(values, weights_obj, corrected=false)

            # Min and max (unweighted since they represent the range)
            min_val = minimum(values)
            max_val = maximum(values)

            push!(variables, col)
            push!(means, weighted_mean)
            push!(medians, weighted_median)
            push!(stds, weighted_std)
            push!(mins, min_val)
            push!(maxs, max_val)
            push!(ess_vals, ess_val)

        elseif col_type <: AbstractArray && eltype(col_type) <: Real
            # Handle vector/array values - compute statistics component-wise and return as vectors

            # Check that all vectors have the same length
            if all(v -> length(v) == length(values[1]), values)
                # Convert to matrix where each row is a particle and each column is a dimension
                value_matrix = hcat([v for v in values]...)'  # Each row is a particle, each column is a dimension
                n_dims = size(value_matrix, 2)

                # Compute statistics for each component
                component_means = Vector{Float64}(undef, n_dims)
                component_medians = Vector{Float64}(undef, n_dims)
                component_stds = Vector{Float64}(undef, n_dims)
                component_mins = Vector{Float64}(undef, n_dims)
                component_maxs = Vector{Float64}(undef, n_dims)

                for dim in 1:n_dims
                    dim_values = value_matrix[:, dim]

                    # Weighted statistics for this dimension
                    component_means[dim] = StatsBase.mean(dim_values, weights_obj)
                    component_medians[dim] = StatsBase.quantile(dim_values, weights_obj, 0.5)
                    component_stds[dim] = StatsBase.std(dim_values, weights_obj, corrected=false)

                    # Min and max for this dimension
                    component_mins[dim] = minimum(dim_values)
                    component_maxs[dim] = maximum(dim_values)
                end

                # Add single row with vector-valued statistics
                push!(variables, col)
                push!(means, component_means)
                push!(medians, component_medians)
                push!(stds, component_stds)
                push!(mins, component_mins)
                push!(maxs, component_maxs)
                push!(ess_vals, ess_val)
            else
                # Vectors have different lengths - skip with warning
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
        ess=ess_vals
    )
end

