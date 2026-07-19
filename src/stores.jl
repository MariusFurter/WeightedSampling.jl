"""
    AbstractParticleStore

Storage backend for particle data. Transformers touch particle columns only
through this interface, so the concrete storage layout is swappable without
touching the algorithm.

# Interface (methods every backend implements)
- `nparticles(store)::Int` — number of particles `N`.
- `hascol(store, name::Symbol)::Bool` — whether column `name` exists.
- `getcol(store, name::Symbol)::AbstractVector` — the column vector.
- `colnames(store)` — iterable of the stored column names (`Symbol`s).
- `broadcast_setcol!(store, name::Symbol, f, args::Tuple)` — the fused
  broadcast-assign primitive `col .= f.(args...)`, creating `col` (length `N`,
  broadcast-inferred element type) if it does not yet exist. This is the only
  write path. It must materialize the broadcast into a destination of length
  `N` so that fusion is preserved even when every element of `args` is a
  scalar/`Ref` (an unfused `f.(Ref(a), Ref(b))` would collapse to a single
  scalar call).
- `resample!(store, indices)` — permute all columns' rows in place, so
  particle `i` becomes old particle `indices[i]`.

Weights are not stored here; they live as a separate `Vector{Float64}` on
[`SMCState`](@ref).

`ColumnStore` is the provided backend.
"""
abstract type AbstractParticleStore end

function nparticles end
function hascol end
function getcol end
function colnames end
function broadcast_setcol! end
function resample! end

# Shared fused-broadcast helper (function barrier: `broadcast!` specializes on
# `col`'s concrete runtime type). The destination `col` (length `N`) drives the
# iteration length, so all-`Ref` argument tuples still produce `N` independent
# results.
#
# For array-valued elements (e.g. `θ .= zeros(J)`, broadcast via a `Ref`),
# `f.(args...)` assigns the SAME array object into every slot of `col`, so
# every particle would alias one underlying array — mutating one particle's
# element in place would corrupt all others. Deep-copying here breaks that
# aliasing immediately.
function _bcast!(col, f, args::Tuple)
    broadcast!(f, col, args...)
    if eltype(col) <: AbstractArray
        col .= copy.(col)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# ColumnStore (ping-pong resampling)
# ---------------------------------------------------------------------------

"""
    ColumnStore(N::Integer)

Particle store backed by a `Dict{Symbol,AbstractVector}` of active columns
(`front`) plus a matching set of persistent scratch buffers (`back`), enabling
allocation-free ("ping-pong") resampling. `names` preserves insertion order.

- Column addition is O(1); existing columns are never reallocated.
- Resampling gathers `front -> back` through the index permutation, then swaps
  the two buffer sets, so no per-column allocation happens after warmup.
"""
mutable struct ColumnStore <: AbstractParticleStore
    n::Int
    names::Vector{Symbol}
    front::Dict{Symbol,AbstractVector}
    back::Dict{Symbol,AbstractVector}
end

ColumnStore(n::Integer) =
    ColumnStore(Int(n), Symbol[], Dict{Symbol,AbstractVector}(), Dict{Symbol,AbstractVector}())

nparticles(s::ColumnStore) = s.n
hascol(s::ColumnStore, name::Symbol) = haskey(s.front, name)
getcol(s::ColumnStore, name::Symbol) = s.front[name]
colnames(s::ColumnStore) = s.names

function broadcast_setcol!(s::ColumnStore, name::Symbol, f, args::Tuple)
    col = get(s.front, name, nothing)
    if col === nothing
        ET = Broadcast.combine_eltypes(f, args)
        col = Vector{ET}(undef, s.n)
        s.front[name] = col
        s.back[name] = Vector{ET}(undef, s.n)
        push!(s.names, name)
    end
    _bcast!(col, f, args)
    return nothing
end

"""
    resample!(s::ColumnStore, indices)

Allocation-free ping-pong resampling: gather each active column into its
persistent scratch buffer, then swap `front`/`back` so the gathered buffers
become active.
"""
function resample!(s::ColumnStore, indices::AbstractVector{<:Integer})
    for name in s.names
        _gather!(s.back[name], s.front[name], indices)
    end
    s.front, s.back = s.back, s.front
    return nothing
end

# Function barrier: dispatch on the concrete column type once, then run the
# permutation type-stably. Array-valued columns are copied so duplicated
# indices don't leave distinct particles aliasing one underlying array.
function _gather!(dest::AbstractVector, src::AbstractVector, indices::AbstractVector{<:Integer})
    @inbounds for i in eachindex(indices)
        dest[i] = src[indices[i]]
    end
    return dest
end

function _gather!(dest::AbstractVector{<:AbstractArray}, src::AbstractVector{<:AbstractArray},
    indices::AbstractVector{<:Integer})
    @inbounds for i in eachindex(indices)
        dest[i] = copy(src[indices[i]])
    end
    return dest
end

"""
    show(io, s::ColumnStore)

Compact one-line summary, e.g. `ColumnStore(n=1000, columns=[:α, :β])`.
"""
Base.show(io::IO, s::ColumnStore) = print(io, "ColumnStore(n=", s.n, ", columns=", s.names, ")")
Base.show(io::IO, ::MIME"text/plain", s::ColumnStore) = show(io, s)
