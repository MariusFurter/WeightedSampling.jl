using Test
using Random


"""
Smoke tests for the custom `Base.show` methods (`ColumnStore`, `SMCState`,
`ParticleTransformer` model trees) that replace Julia's default struct
printing (see docstrings in `src/stores.jl`/`src/types.jl`/`src/transformers.jl`).
Just checks the compact and `MIME"text/plain"` forms run without error and
contain the expected key fields — not exhaustive formatting checks.
"""
@model function normal_model_show(μ, σ)
    x ~ Normal(μ, σ)
end

@testset "ColumnStore show" begin
    store = ColumnStore(100)
    compact = sprint(show, store)
    @test compact == "ColumnStore(n=100, columns=Symbol[])"

    verbose = sprint(io -> show(io, MIME"text/plain"(), store))
    @test verbose == compact
end

@testset "SMCState show" begin
    Random.seed!(42)
    state = SMCState(100)
    model = normal_model_show(0.0, 1.0; kernels=(Normal=NormalKernel,))
    apply!(model, state)

    compact = sprint(show, state)
    @test occursin("SMCState(n_particles=100", compact)
    @test occursin(":x", compact)

    verbose = sprint(io -> show(io, MIME"text/plain"(), state))
    @test occursin("n_particles:  100", verbose)
    @test occursin("ess_perc_min:", verbose)
    @test occursin("resampled:", verbose)
    @test occursin("depth:", verbose)
end

@testset "ParticleTransformer show (compact + tree)" begin
    model = normal_model_show(0.0, 1.0; kernels=(Normal=NormalKernel,))

    compact = sprint(show, model)
    @test occursin("Sequence", compact)

    tree = sprint(io -> show(io, MIME"text/plain"(), model))
    @test occursin("Sequence", tree)
    @test occursin("Sample(:x)", tree)
end
