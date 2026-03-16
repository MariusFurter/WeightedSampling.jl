using WeightedSampling
using Test
using Random
using Distributions

include("closed_form_conditioning.jl")
include("normal_normal_test.jl")
include("random_walk_test.jl")
include("kalman_evidence_test.jl")
include("linear_regression_mh_test.jl")

@testset "WeightedSampling.jl" begin
    @testset "Beta-Binomial conjugate" begin
        @test beta_binomial_test(10, 8, 10, 1.0, 2.0)
    end

    @testset "Normal-Normal conjugate" begin
        @test normal_normal_test()
    end

    @testset "Sequential random walk" begin
        @test random_walk_test()
    end

    @testset "Kalman filter log-evidence" begin
        @test kalman_evidence_test()
    end

    @testset "Linear regression with MH moves" begin
        @test linear_regression_mh_test()
    end
end
