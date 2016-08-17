using Base.Test
using Distributions
using HiddenMarkovModels

println("hi")

# include("hmm_discrete.jl") #Alex
# include("hmm_normal.jl")   #Alex

include("ddm.jl")
include("dhmm.jl")
include("rkhs_tupletype.jl")
include("rkhs_vptree.jl")
include("rkhs_types.jl")
include("rkhs_filtering.jl")
