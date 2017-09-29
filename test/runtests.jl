using Base.Test
using HiddenMarkovModels
HMM = HiddenMarkovModels



println()
println("* Testing legacy back-end...")
include("discrete-with-old-backend.jl")   # requires ForwardDiff and Calculus

# println()
# println("* Testing legacy back-end...")
# include("utils.jl")

println()
println("* Testing discrete filtering...")
include("discrete.jl")

println()
println("* Testing Kalman filtering...")
include("linear-gaussian.jl")
