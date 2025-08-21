# Bonus: Add check that the same variable is not used multiple times. E.g keep a list of variables used in sampling statements up until now and check that the current variable is not in that list.

### Future: Think about adding pseudo-marginal sampling when some variables are overwritten.

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
            rhs_vars = reduce(union, map(extract_symbols, args), init=Set{Symbol}())
            vars = union(lhs_vars, rhs_vars)
            vars = setdiff(vars, exceptions)

            out_getter, _ = capture_lhs(lhs)
            lhs_rewritten = out_getter(particles_sym)

            args_rewritten = map(args) do arg
                replace_symbols_except(arg, exceptions, particles_sym, N_sym)
            end

            e = quote
                if tracker <= target_depth && !isempty(intersect(targets, $vars))
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
            rhs_vars = reduce(union, map(extract_symbols, args), init=Set{Symbol}())
            vars = union(lhs_vars, rhs_vars)
            vars = setdiff(vars, exceptions)

            lhs_rewritten = replace_symbols_except(lhs, exceptions, particles_sym, N_sym)
            args_rewritten = map(args) do arg
                replace_symbols_except(arg, exceptions, particles_sym, N_sym)
            end

            e = quote
                if tracker <= target_depth && !isempty(intersect(targets, $vars))
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
        function smc_logpdf($particles_sym, targets, target_depth)

            N = DrawingInferences.nrow($particles_sym)

            scores = zeros(N)
            tracker = 1

            $(build_logpdf_body(body, exceptions, particles_sym, N_sym))

            return scores
        end
    end
end

function mh!(particles, proposal_kernel, targets, target_depth, kernel_args, logpdf_fun)
    # Calculate log of MH ratio
    # r = p(x_new)q(x_old | x_new) / p(x_old)q(x_new | x_old)

    # log_probs = log q(x_old | x_new) - log q(x_new | x_old)
    changes, log_probs = proposal_kernel(particles, targets, kernel_args...)

    # log_probs -= log p(x_old)
    log_probs -= logpdf_fun(particles, targets, target_depth)

    old_values = merge_particles!(particles, changes)

    # log_probs +- log p(x_new)
    log_probs += logpdf_fun(particles, targets, target_depth)

    # Accept changes w. prob min(1, r)
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
