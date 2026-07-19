using Test
using Random
using Statistics
using DataFrames


"""
Covers the convenience/analysis API surface in `src/utils.jl`/`src/types.jl`
that no other test file exercises: `SMCState(n)`, `state[:x]`, `@E`,
`DataFrame(state)`, `sample(state, n)`, `describe(state)`.
"""
@model function normal_model_api(μ, σ)
    x ~ Normal(μ, σ)
end

@testset "SMCState(n) convenience constructor and state[:x]" begin
    Random.seed!(42)
    N = 10_000
    state = SMCState(N)
    model = normal_model_api(1.0, 2.0; kernels=(Normal=NormalKernel,))
    apply!(model, state)

    @test state[:x] === getcol(state.store, :x)
    @test length(state[:x]) == N
end

@testset "@E computes weighted expectations" begin
    Random.seed!(42)
    N = 200_000
    state = SMCState(N)
    model = normal_model_api(2.0, 3.0; kernels=(Normal=NormalKernel,))
    apply!(model, state)

    @test isapprox(@E(x -> x, state), 2.0, atol=0.05)
    @test isapprox(@E(x -> x^2, state), 2.0^2 + 3.0^2, atol=0.2)
end

@testset "DataFrame(state) exports all particles with log_weight" begin
    Random.seed!(42)
    N = 1_000
    state = SMCState(N)
    model = normal_model_api(0.0, 1.0; kernels=(Normal=NormalKernel,))
    apply!(model, state)

    df = DataFrame(state)
    @test nrow(df) == N
    @test df.x == getcol(state.store, :x)
    @test df.log_weight == state.weights
end

@testset "sample(state, n) draws a weighted resample" begin
    Random.seed!(42)
    N = 10_000
    state = SMCState(N)
    model = normal_model_api(0.0, 1.0; kernels=(Normal=NormalKernel,))
    apply!(model, state)

    df = sample(state, 500)
    @test nrow(df) == 500
    @test isapprox(mean(df.x), 0.0, atol=0.2)

    @test_throws ArgumentError sample(state, 0)
    @test_throws ArgumentError sample(state, N + 1; replace=false)
end

@testset "describe(state) returns weighted summary statistics" begin
    Random.seed!(42)
    N = 100_000
    state = SMCState(N)
    model = normal_model_api(5.0, 2.0; kernels=(Normal=NormalKernel,))
    apply!(model, state)

    summary = describe(state)
    @test summary.variable == [:x]
    @test isapprox(only(summary.mean), 5.0, atol=0.1)
    @test isapprox(only(summary.std), 2.0, atol=0.1)
    @test only(summary.hist) isa String
    @test !isempty(only(summary.hist))
end
