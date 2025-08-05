using MacroTools
using Distributions
using Random
using BenchmarkTools

function replace_symbols(expr, particle_sym, exceptions)
    if expr isa Symbol && !(expr in exceptions)
        return :(getproperty($particle_sym, $(QuoteNode(expr))))
    elseif expr isa Expr
        if expr.head == :call
            return Expr(:call, expr.args[1], map(x -> replace_symbols(x, particle_sym, exceptions), expr.args[2:end])...)
        else

            return Expr(expr.head, map(x -> replace_symbols(x, particle_sym, exceptions), expr.args)...)
        end
    else
        return expr
    end
end

function capture_lhs(expr)

    if expr isa Symbol
        output_expr = QuoteNode(expr)
    elseif @capture(expr, output_symbol_{index_})
        output_expr = :(Symbol($(QuoteNode(output_symbol)), $index))
    else
        error("Left-hand side must be a variable name `x` or an indexed variable `x{i}`")
    end

    return output_expr
end

function rewrite_assignment(expr, exceptions)
    @capture(expr, lhs_ = rhs_)

    output_expr = capture_lhs(lhs)

    particle_sym = gensym("particle")
    i_sym = gensym("i")
    rhs_rewritten = replace_symbols(rhs, particle_sym, exceptions)

    quote
        let output = $output_expr
            for ($i_sym, $particle_sym) in enumerate(particles.samples)
                val = $(rhs_rewritten)
                setproperty!($particle_sym, output, val)
                # new_var = NamedTuple{(output,)}((val,))
                # particles.samples[$i_sym] = merge($particle_sym, new_var)
            end
        end
    end
end

function rewrite_sampling(expr, exceptions)
    @capture(expr, lhs_ ~ f_(args__))

    output_expr = capture_lhs(lhs)

    particle_sym = gensym("particle")
    i_sym = gensym("i")
    args_rewritten = map(args) do arg
        replace_symbols(arg, particle_sym, exceptions)
    end

    quote
        let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                kernels.$f
            else
                $f
            end
            output = $output_expr
            for ($i_sym, $particle_sym) in enumerate(particles.samples)
                val = kernel.sampler($(args_rewritten...), rng)
                setproperty!($particle_sym, output, val)
                # new_var = NamedTuple{(output,)}((val,))
                # particles.samples[$i_sym] = merge($particle_sym, new_var)
                if kernel.weighter !== nothing
                    particles.weights[$i_sym] += kernel.weighter($(args_rewritten...), val)
                end
            end
        end
    end
end

function rewrite_observe(expr, exceptions)
    @capture(expr, lhs_ -> f_(args__))

    particle_sym = gensym("particle")
    i_sym = gensym("i")
    lhs_rewritten = replace_symbols(lhs, particle_sym, exceptions)
    args_rewritten = map(args) do arg
        replace_symbols(arg, particle_sym, exceptions)
    end

    quote
        let kernel = if hasproperty(kernels, $(QuoteNode(f)))
                kernels.$f
            else
                $f
            end
            for ($i_sym, $particle_sym) in enumerate(particles.samples)
                val = $lhs_rewritten
                particles.weights[$i_sym] += kernel.logpdf($(args_rewritten...), val)
            end
        end
    end
end

function build_smc(body, exceptions)
    code = quote end
    for statement in body

        if @capture(statement, lhs_ = rhs_)
            rewritten_statement = rewrite_assignment(statement, exceptions)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, lhs_ ~ f_(args__))
            rewritten_statement = rewrite_sampling(statement, exceptions)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, lhs_ -> f_(args__))
            rewritten_statement = rewrite_observe(statement, exceptions)
            append!(code.args, rewritten_statement.args)

        elseif @capture(statement, for loop_var_ in start_:stop_
            loop_body__
        end)

            push!(exceptions, loop_var)
            e = quote
                for $loop_var in $start:$stop
                    $(build_smc(loop_body, exceptions))
                end
            end
            delete!(exceptions, loop_var)
            append!(code.args, e.args)
        end
    end
    return code
end

macro parse(expr)
    @capture(expr, function name_()
        body__
    end) || error("Expression must be a function definition")

    exceptions = Set{Symbol}()

    dummy_sym = gensym("dummy")
    name! = Symbol(name, "!")
    particle_name_sym = gensym(Symbol(name, "Particle"))

    # Later: replace with default kernels provided by DrawingInferences
    return esc(quote
        # Mutate existing particles 
        function $name!(particles, kernels=nothing, rng=Random.default_rng())
            $(build_smc(body, exceptions))
        end

        # Run with a single particle
        function $name(kernels=nothing, rng=Random.default_rng())
            $dummy_sym = SMCParticles([DummyParticle()], [0.0])
            $name!($dummy_sym, kernels, rng)
            return $dummy_sym
        end

        # Run create new particles and run
        function $name(N_particles::Int64, kernels=nothing, rng=Random.default_rng())
            $dummy_sym = $name(kernels, rng)
            variables = map(collect($dummy_sym.samples[1].dict)) do (key, value)
                (key => typeof(value))
            end |> Dict
            initial_values = values($dummy_sym.samples[1].dict)
            make_struct($(QuoteNode(particle_name_sym)), variables)
            particle_type = $particle_name_sym(initial_values...)
            particles = SMCParticles([particle_type for _ in 1:N_particles], [0.0 for _ in 1:N_particles])
            $name!(particles, kernels, rng)
            return particles
        end
    end)
end

struct SMCParticles{P}
    samples::Vector{P}
    weights::Vector{Float64}
end

Base.show(io::IO, particles::SMCParticles) = print(io, "SMCParticles with $(length(particles.samples)) samples")
Base.show(io::IO, ::MIME"text/plain", particles::SMCParticles) = println(io, "SMCParticles with $(length(particles.samples)) samples")

struct SMCKernel{S,L,W}
    sampler::S
    logpdf::L
    weighter::W
end

struct DummyParticle
    dict::Dict{Symbol,Any}
    function DummyParticle()
        new(Dict{Symbol,Any}())
    end
end

function Base.getproperty(p::DummyParticle, key::Symbol)
    if key === :dict
        getfield(p, :dict)
    else
        get(getfield(p, :dict), key, nothing)
    end
end
Base.setproperty!(p::DummyParticle, key::Symbol, value) = (getfield(p, :dict)[key] = value)


function make_struct(name, variables)
    fields = [:($field::$type) for (field, type) in variables]
    struct_code = quote
        mutable struct $name
            $(fields...)
        end
    end
    Core.eval(Main, struct_code)
end

initialKernel = SMCKernel(
    (rng) -> randn(rng),
    (x) -> logpdf(Normal(0, 1), x),
    nothing
)

walkKernel = SMCKernel(
    (x_in::Float64, rng) -> x_in + randn(rng),
    (x_in::Float64, x_out::Float64) -> logpdf(Normal(x_in, 1), x_out),
    (x_in::Float64, x_out::Float64) -> 0.0
)

my_kernels = (initialKernel=initialKernel, walkKernel=walkKernel)

my_rng = Random.default_rng()

@parse function bla()
    x0 ~ initialKernel()
    for i in 1:10_000
        x{i} ~ walkKernel(x0)
    end
end

@btime bla(1000, my_kernels, my_rng)

### 100 time-steps + single variable
### 10^5 particles: 46ms / 450 allocs


### 100 time-steps + dynamic variables
### 10^3 particles: 19ms / 100k allocs
### 10^4 particles: 72ms / 1M allocs
### 10^5 particles: 600ms / 10M allocs
### 10^6 particles: 6s / 100M allocs

### 10k time-steps + single variable
### 10^2 particles: 5ms / 10k allocs
### 10^4 particles: 450ms / 10k allocs

### 10k time-steps + dynamic variables
### 10^2 particles: 21s / 2M allocs
### 10^3 particles: 51s / 10M allocs

## The issue of dynamically create structs surfaces again... Works if called twice.

## ToDo: Dynamic indices in args