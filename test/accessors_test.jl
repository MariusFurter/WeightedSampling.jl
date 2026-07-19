using Test
using Random


"""
Value-level accessor test models: `[]` (getindex) and `.` (getproperty)
reads/writes, including a chained case. `v`/`bag` are array-/struct-valued
particle columns built via a plain `:vect`/constructor `Assign` (no custom
kernel needed) so these tests isolate `vectorize`'s accessor recursion and
`AccessorAssign`'s in-place mutation.
"""

mutable struct Point
    x::Float64
    y::Float64
end

mutable struct Bag
    v::Vector{Float64}
end

# `[]` reads and writes on an array-valued column.
@model function idx_model()
    a ~ Normal(0, 1)
    b ~ Normal(10, 1)
    v .= [a, b]
    s .= v[1] + v[2]
    v[1] .= v[1] + 100.0
end

# `.` reads and writes on a mutable-struct-valued column.
@model function prop_model()
    a ~ Normal(0, 1)
    b ~ Normal(10, 1)
    p .= Point(a, b)
    s .= p.x + p.y
    p.x .= p.x + 100.0
end

# Chained accessor: `.` then `[]`, both directions.
@model function chained_model()
    a ~ Normal(0, 1)
    b ~ Normal(10, 1)
    bag .= Bag([a, b])
    s .= bag.v[1] + bag.v[2]
    bag.v[1] .= bag.v[1] + 100.0
end

function idx_accessor_test(; N=1000)
    Random.seed!(42)
    model = idx_model(; kernels=(Normal=NormalKernel,))
    state = SMCState(ColumnStore(N))
    apply!(model, state)

    a = copy(getcol(state.store, :a))
    b = getcol(state.store, :b)
    s = getcol(state.store, :s)
    v = getcol(state.store, :v)

    read_ok = all(isequal.(s, a .+ b))
    # after the write, v[i][1] should equal a[i] + 100 (the pre-write a value)
    write_ok = all(i -> v[i][1] == a[i] + 100.0, eachindex(v))
    other_untouched = all(i -> v[i][2] == b[i], eachindex(v))

    return read_ok && write_ok && other_untouched
end

function prop_accessor_test(; N=1000)
    Random.seed!(42)
    model = prop_model(; kernels=(Normal=NormalKernel,))
    state = SMCState(ColumnStore(N))
    apply!(model, state)

    a = copy(getcol(state.store, :a))
    b = getcol(state.store, :b)
    s = getcol(state.store, :s)
    p = getcol(state.store, :p)

    read_ok = all(isequal.(s, a .+ b))
    write_ok = all(i -> p[i].x == a[i] + 100.0, eachindex(p))
    other_untouched = all(i -> p[i].y == b[i], eachindex(p))

    return read_ok && write_ok && other_untouched
end

function chained_accessor_test(; N=1000)
    Random.seed!(42)
    model = chained_model(; kernels=(Normal=NormalKernel,))
    state = SMCState(ColumnStore(N))
    apply!(model, state)

    a = copy(getcol(state.store, :a))
    b = getcol(state.store, :b)
    s = getcol(state.store, :s)
    bag = getcol(state.store, :bag)

    read_ok = all(isequal.(s, a .+ b))
    write_ok = all(i -> bag[i].v[1] == a[i] + 100.0, eachindex(bag))
    other_untouched = all(i -> bag[i].v[2] == b[i], eachindex(bag))

    return read_ok && write_ok && other_untouched
end

@testset "Value-level accessors ([], ., chains)" begin
    @test idx_accessor_test()
    @test prop_accessor_test()
    @test chained_accessor_test()
end
