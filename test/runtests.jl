using Base.Test
using HiddenMarkovModels

println("hi")

include("utils.jl")

# Alex
# include("hmm_discrete.jl") 
# include("hmm_normal.jl")

# DMM back-end
# include("ddm.jl")

# Discrete HMMs
# include("dhmm.jl")

include("rkhs_tupletype.jl")
include("rkhs_vptree.jl")
include("rkhs_types.jl")
include("rkhs_filtering.jl")

# include("drkhs.jl")
