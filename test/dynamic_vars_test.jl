using Test
using Random
using Statistics


const dyn_kernels = (Normal=NormalKernel,)

"""
Dynamic-variable family test model: retains the full random-walk trajectory
as columns `x_1, x_2, ..., x_{T+1}` (rather than overwriting a single `x`
column), via `x{1} ~ ...` / `x{t+1} ~ Normal(x{t}, 1)`.
"""
@model function dynamic_rw(T::Int)
    x{1} ~ Normal(0, 1)
    for t in 1:T
        x{t + 1} ~ Normal(x{t}, 1)
    end
end

function dynamic_rw_correctness_test(; T=10, N=100_000, atol=0.15)
    Random.seed!(42)
    model = dynamic_rw(T; kernels=dyn_kernels)
    state = SMCState(ColumnStore(N))
    apply!(model, state)

    # Every trajectory column x_1..x_{T+1} must exist and be retained.
    all_cols_present = all(hascol(state.store, Symbol(:x, :_, t)) for t in 1:(T+1))

    final = getcol(state.store, Symbol(:x, :_, T + 1))
    true_var = T + 1
    mean_ok = isapprox(mean(final), 0.0, atol=atol)
    var_ok = isapprox(var(final), true_var, rtol=0.05)

    # Each step's increment x_{t+1} - x_t should be ~ Normal(0, 1) independent
    # of the trajectory so far; spot-check step 1's increment variance.
    x1 = getcol(state.store, :x_1)
    x2 = getcol(state.store, :x_2)
    step_var_ok = isapprox(var(x2 .- x1), 1.0, rtol=0.1)

    return all_cols_present && mean_ok && var_ok && step_var_ok
end

@testset "Dynamic-variable families x{i}" begin
    @test dynamic_rw_correctness_test()
end

@testset "Dynamic-variable error paths" begin
    # Index must not depend on a particle variable.
    @test_throws Exception eval(:(@model function bad_dyn_idx()
        y ~ Normal(0, 1)
        x{y} ~ Normal(0, 1)
    end))

    # A base symbol cannot be both a plain particle variable and a
    # dynamic-variable family (either order).
    @test_throws Exception eval(:(@model function bad_collision1()
        x ~ Normal(0, 1)
        x{1} ~ Normal(0, 1)
    end))

    @test_throws Exception eval(:(@model function bad_collision2()
        x{1} ~ Normal(0, 1)
        x ~ Normal(0, 1)
    end))

    # Reading a dynamic family before it has ever been assigned.
    @test_throws Exception eval(:(@model function bad_unregistered()
        y .= x{1} + 1
    end))
end

"""
Chaining `[]`/`.` onto a dynamic-variable family: `v{1}` is an array-valued
dynamic family member, read/written through `[]`; `p{1}` is a
mutable-struct-valued dynamic family member, read/written through `.`. Mirrors
`accessors_test.jl`'s `idx_model`/`prop_model`, but with a dynamic-family base
instead of a plain particle variable.
"""
mutable struct DynPoint
    x::Float64
    y::Float64
end

@model function dyn_idx_model()
    a ~ Normal(0, 1)
    b ~ Normal(10, 1)
    v{1} .= [a, b]
    s .= v{1}[1] + v{1}[2]
    v{1}[1] .= v{1}[1] + 100.0
end

@model function dyn_prop_model()
    a ~ Normal(0, 1)
    b ~ Normal(10, 1)
    p{1} .= DynPoint(a, b)
    s .= p{1}.x + p{1}.y
    p{1}.x .= p{1}.x + 100.0
end

function dyn_idx_accessor_test(; N=1000)
    Random.seed!(42)
    model = dyn_idx_model(; kernels=dyn_kernels)
    state = SMCState(ColumnStore(N))
    apply!(model, state)

    a = copy(getcol(state.store, :a))
    b = getcol(state.store, :b)
    s = getcol(state.store, :s)
    v = getcol(state.store, :v_1)

    read_ok = all(isequal.(s, a .+ b))
    write_ok = all(i -> v[i][1] == a[i] + 100.0, eachindex(v))
    other_untouched = all(i -> v[i][2] == b[i], eachindex(v))

    return read_ok && write_ok && other_untouched
end

function dyn_prop_accessor_test(; N=1000)
    Random.seed!(42)
    model = dyn_prop_model(; kernels=dyn_kernels)
    state = SMCState(ColumnStore(N))
    apply!(model, state)

    a = copy(getcol(state.store, :a))
    b = getcol(state.store, :b)
    s = getcol(state.store, :s)
    p = getcol(state.store, :p_1)

    read_ok = all(isequal.(s, a .+ b))
    write_ok = all(i -> p[i].x == a[i] + 100.0, eachindex(p))
    other_untouched = all(i -> p[i].y == b[i], eachindex(p))

    return read_ok && write_ok && other_untouched
end

@testset "[]/. chained onto a dynamic-variable family x{i}" begin
    @test dyn_idx_accessor_test()
    @test dyn_prop_accessor_test()
end

@testset "Chained dynamic-variable accessor error paths" begin
    # Chaining onto a dynamic family that was never assigned.
    @test_throws Exception eval(:(@model function bad_dyn_chain_unregistered()
        v{1}[1] .= 1.0
    end))

    # The index-purity rule still applies inside a chained dynamic accessor.
    @test_throws Exception eval(:(@model function bad_dyn_chain_idx()
        y ~ Normal(0, 1)
        v{1} .= [1.0, 2.0]
        v{y}[1] .= 1.0
    end))
end
