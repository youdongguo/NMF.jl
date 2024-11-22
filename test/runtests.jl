using NMF
using Test
using Random
using LinearAlgebra
using StatsBase
using ForwardDiff

include("testproblems.jl")

tests = ["utils",
         "initialization",
         "spa",
         "multupd",
         "alspgrad",
         "coorddesc",
         "greedycd",
         "interf",
         "weightedcoorddesc"]

println("Running tests:")
@testset "All tests" begin
    for t in tests
        tp = "$t.jl"
        include(tp)
    end
end
