using DrawingInferences
using Test
using Random
using Distributions

include("closed_form_conditioning.jl")

@testset "DrawingInferences.jl" begin
    @test beta_binomial_test(10, 8, 10, 1.0, 2.0)
end
