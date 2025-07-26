using SequentialMonteCarlo
using RNGPool

struct FKStep{I,S,W}
    inputs::I
    output::Symbol
    output_type::DataType
    sampler::S
    weighter::W
end

### FKModel
struct FKModel{T<:Union{AbstractVector{<:FKStep},Tuple{Vararg{FKStep}}}}
    steps::T

    function FKModel(steps::T) where {T<:Union{AbstractVector{<:FKStep},Tuple{Vararg{FKStep}}}}
        validate_step_dependencies(steps)
        new{T}(steps)
    end
end

function validate_step_dependencies(steps)
    available_outputs = Set{Symbol}()

    for (i, step) in enumerate(steps)
        # Check that all inputs for this step are available from previous steps
        missing_inputs = setdiff(Set(step.inputs), available_outputs)

        if !isempty(missing_inputs)
            missing_str = join(missing_inputs, ", ")
            error("FKModel validation error: Step $i uses undefined variables: $missing_str. " *
                  "These variables must be outputs of previous steps.")
        end
        # Add this step's output to the available outputs
        push!(available_outputs, step.output)
    end
end

Base.length(fk::FKModel) = length(fk.steps)
Base.getindex(fk::FKModel, i::Int) = fk.steps[i]
Base.show(io::IO, fk::FKModel) = print(io, "FKModel($(length(fk)) steps)")
function Base.show(io::IO, ::MIME"text/plain", fk::FKModel)
    vars = variables(fk)
    vars_str = isempty(vars) ? "none" : join(vars, ", ")
    println(io, "FKModel with $(length(fk)) steps")
    println(io, "Variables: $vars_str")
end
Base.iterate(fk::FKModel) = iterate(fk.steps)
Base.iterate(fk::FKModel, state) = iterate(fk.steps, state)

function variables(fk::FKModel)
    all_vars = Set{Symbol}()
    for step in fk.steps
        union!(all_vars, step.inputs)
        push!(all_vars, step.output)
    end
    return sort(collect(all_vars))
end

### SMC Model creation

function initial_value(t::DataType)
    if t === Any
        return missing
    else
        try
            return zero(t)
        catch e
            error("Initial values for type $t not implemented.")
        end
    end
end

mutable struct GenericParticle{T<:NamedTuple}
    p::T

    # Initializer for GenericParticle
    function GenericParticle{T}() where {T<:NamedTuple}
        num_fields = fieldcount(T)
        initial_values = Tuple(initial_value(fieldtype(T, i)) for i in 1:num_fields)
        initial_namedtuple = T(initial_values)
        return new{T}(initial_namedtuple)
    end

    GenericParticle(p::T) where {T<:NamedTuple} = new{typeof(p)}(p)
end

Base.getindex(particle::GenericParticle, key::Symbol) = getproperty(particle.p, key)

function SequentialMonteCarlo.SMCModel(fk::FKModel)
    T = length(fk)

    # Collect all unique variables and their types from the model
    var_map = Dict(step.output => step.output_type for step in fk.steps)

    vars = Tuple(variables(fk))
    types = Tuple(var_map[v] for v in vars)

    # Create a template for the particle's NamedTuple
    particle_type = NamedTuple{vars,Tuple{types...}}
    ParticleType = GenericParticle{particle_type}

    sampler_lookup = Tuple(step.sampler for step in fk.steps)

    function M!(newParticle, rng::RNG, p::Int64, particle, ::Nothing)
        if sampler_lookup[p] !== nothing
            newParticle.p = sampler_lookup[p](particle.p, rng)
        end
    end

    weighter_lookup = Tuple(step.weighter for step in fk.steps)

    function lG(p::Int64, particle, ::Nothing)
        if weighter_lookup[p] === nothing
            return 0.0
        else
            return weighter_lookup[p](particle.p)
        end
    end

    return SMCModel(M!, lG, T, ParticleType, Nothing)
end


# Custom show methods for SMCModel to avoid verbose type printing
Base.show(io::IO, model::SequentialMonteCarlo.SMCModel) = print(io, "SMCModel(T=$(model.maxn))")

function Base.show(io::IO, ::MIME"text/plain", model::SequentialMonteCarlo.SMCModel)
    println(io, "SMCModel:")
    println(io, "  Time steps: $(model.maxn)")
    println(io, "  Particle type: $(model.particle)")
    println(io, "  Scratch type: $(model.pScratch)")
end

# Custom show methods for SMCIO to avoid verbose type printing
Base.show(io::IO, smcio::SequentialMonteCarlo.SMCIO) = print(io, "SMCIO(N=$(smcio.N), T=$(smcio.maxn))")

function Base.show(io::IO, ::MIME"text/plain", smcio::SequentialMonteCarlo.SMCIO)
    println(io, "SMCIO:")
    println(io, "  Particles: $(smcio.N)")
    println(io, "  Time steps: $(smcio.n)")
    println(io, "  Threads: $(smcio.nthreads)")
    println(io, "  Full output: $(smcio.fullOutput)")
    println(io, "  ESS threshold: $(smcio.essThreshold)")
end