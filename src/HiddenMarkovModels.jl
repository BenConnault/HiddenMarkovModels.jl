module HiddenMarkovModels

importall DynamicDiscreteModels
importall Distributions
using StatsBase:sample,WeightVec,StatisticalModel

# When I get around to spinning of a Markov.jl
# import Markov: rsm, nsm, z2q, q2z
include("stochasticmatrices.jl")

# Ben's exports
export 	coef!, rand, loglikelihood, mle, dim, 
		em, viterbi,
		baumwelch, hmm, theta2ab

#Alex's exports
export	HMM, forward_backward, viterbi, smoothed_forward_backward, fit!


#Ben's sources
include("hiddenmarkovmodel.jl")

#Alex's sources
include("HMM.jl")
include("forward_backward.jl")
include("fit.jl")




end