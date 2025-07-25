using SequentialMonteCarlo
using RNGPool

struct WeightedSampler{S,W,O}
    sampler::S
    weighter::W
    output_type::O
end

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

mutable struct GenericParticle{T<:NamedTuple}
    p::T
    GenericParticle{T}() where {T<:NamedTuple} = new{T}(T(Tuple(zero(fieldtype(T, i)) for i in 1:fieldcount(T))))
end


# ToDo: Make work for arbitrary variables
# Then: Examples + Convenience functions
# ToDo: Variable interpolation in loops.
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
        newParticle.p = sampler_lookup[p](particle.p, rng)
    end

    weighter_lookup = Tuple(step.weighter for step in fk.steps)

    function lG(p::Int64, particle, ::Nothing)
        return weighter_lookup[p](particle.p)
    end

    return SMCModel(M!, lG, T, ParticleType, Nothing)
end


### Custom show methods for SMCModel to avoid verbose type printing
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