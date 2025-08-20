# Bonus: Add check that the same variable is not used multiple times. E.g keep a list of variables used in sampling statements up until now and check that the current variable is not in that list.

## Have logpdf_difference operate on current_particles and proposed_changes (that only includes diffs as cols)

## scores .+= kernel.logpdf.(args_rewritten, value_rewritten)
## args_rewritten:
## x -> if :x in diffs proposed_changes[!,x] else current_particles[!,x]

function build_logpdf_body(body, exceptions, particles_sym, N_sym)
    code = quote end

    for statement in body

        if @capture(statement, lhs_ = rhs_) || @capture(statement, lhs_ << f_(args__))
            e = quote
                tracker += 1
            end
            append!(code.args, e.args)

        elseif @capture(statement, lhs_ ~ f_(args__))
            lhs_vars = extract_symbols(lhs)
            rhs_vars = reduce(union, map(extract_symbols, args))
            vars = union(lhs_vars, rhs_vars)
            vars = setdiff(vars, exceptions)

            out_getter, _ = capture_lhs(lhs)
            lhs_rewritten = out_getter(particles_sym)

            args_rewritten = map(args) do arg
                replace_symbols_except(arg, exceptions, particles_sym, N_sym)
            end

            e = quote
                if tracker <= target_depth && !isempty(intersect(diffs, $vars))
                    let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                            kernels.$f
                        else
                            $f
                        end
                        scores .+= kernel.logpdf.($(args_rewritten...), $lhs_rewritten)
                    end
                end
                tracker += 1
            end

            append!(code.args, e.args)

        elseif @capture(statement, lhs_ => f_(args__))
            lhs_vars = extract_symbols(lhs)
            rhs_vars = reduce(union, map(extract_symbols, args))
            vars = union(lhs_vars, rhs_vars)
            vars = setdiff(vars, exceptions)

            lhs_rewritten = replace_symbols_except(lhs, exceptions, particles_sym, N_sym)
            args_rewritten = map(args) do arg
                replace_symbols_except(arg, exceptions, particles_sym, N_sym)
            end

            e = quote
                if tracker <= target_depth && !isempty(intersect(diffs, $vars))
                    let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                            kernels.$f
                        else
                            $f
                        end
                        scores .+= kernel.logpdf.($(args_rewritten...), $lhs_rewritten)
                    end
                end
                tracker += 1
            end

            append!(code.args, e.args)

        elseif @capture(statement, for loop_var_ in start_:stop_
            loop_body__
        end)

            push!(exceptions, loop_var)
            e = quote
                for $loop_var in $start:$stop
                    $(build_logpdf_body(loop_body, exceptions, particles_sym, N_sym))
                end
            end
            delete!(exceptions, loop_var)
            append!(code.args, e.args)

        else
            error("Unsupported statement type: $statement")
        end
    end

    return code
end

function build_logpdf(body, exceptions, N_sym)

    particles_sym = :particles

    return quote
        function logpdf($particles_sym, diffs, target_depth)

            N = DrawingInferences.nrow($particles_sym)

            scores = zeros(N)
            tracker = 1

            $(build_logpdf_body(body, exceptions, particles_sym, N_sym))

            return scores
        end
    end
end

## Proposal kernel should take particles and var_symbols as input
## mutate the particles in place, and return a vector of log scores.
## (particles, in_symbols, out_symbols) -> log_scores

## This is general enough to allow for adaptive proposals.
## In fact, one could also use this signature for SMCKernels.
## No: does not seem to support partial evaluation with literals well.

function mh!(proposal_kernel, targets, kernel_args, particles)

    proposed_changes, log_probs = proposal_kernel(particles, targets, kernel_args...)

    log_probs -= logpdf(particles, diffs, target_depth)

    old_values = merge_particles!(particles, proposed_changes)

    log_probs += logpdf(particles, diffs, target_depth)

    rejected_changes = reject(log_probs)

    merge_particles!(particles, old_values, rejected_changes)
    return nothing
end

function merge_particles!(particles, changes, positions=nothing)

    if positions === nothing
        positions = fill(true, nrow(particles))
    end

    targets = names(changes)
    old_values = particles[:, targets]

    for col in names(changes)
        if eltype(particles[!, col]) <: AbstractVector
            particles[positions, col] .= copy.(changes[positions, col])
        else
            particles[positions, col] .= changes[positions, col]
        end
    end

    return old_values
end

reject(log_probs) = log.(rand(length(log_probs))) .>= log_probs


function RW!(particles, targets)
    # Gaussian RW with covariance λ Σ
    # where λ = 2.38 d^-1/2 and Σ is the empirical covariance matrix of the target particles
    # targets :: Vector
    N = nrow(particles)
    d = length(targets)
    λ = 2.38 * d^(-1 / 2)

    m = Matrix(particles[!, targets])
    w = ProbabilityWeights(exp_norm_weights(particles[!, :weights]))
    Σ = cov(m, w)

    # Replace 0 values with minimum step epsilon
    eps = 1e-3
    Σ[Σ.==0] .= eps

    changes = rand(MvNormal(λ * Σ), N)

    for (i, col) in enumerate(targets)
        changes[i, :] .+= particles[!, col]
    end

    return DataFrame(changes', targets)
end

### Future: Think about adding pseudo-marginal sampling when some variables are overwritten.