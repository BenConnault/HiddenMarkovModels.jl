module HiddenMarkovModels

importall DynamicDiscreteModels
importall Distributions
using StatsBase:sample,WeightVec,StatisticalModel

# When I get around to spinning of a Markov.jl
# import Markov: rsm, nsm, z2q, q2z
include("utils_stochasticmatrices.jl")

# Ben's exports
export 	coef!, rand, loglikelihood, mle, dim, 
		em, viterbi, filtr,
		baumwelch, hmm, theta2ab

#Alex's exports
export	HMM, forward_backward, viterbi, smoothed_forward_backward, fit!


#Ben's sources
include("dhmm.jl")

#Alex's sources
include("hmm_types.jl")
include("hmm_filtering.jl")
include("hmm_fit.jl")

#############################################
# RKHS branch
import Base.transpose

include("rkhs_types.jl")
include("rkhs_calculus.jl")
include("rkhs_filtering.jl")

export line
export RKHSLeftElement, RKHSRightElement, RKHSMap, RKHS2, marginal, marginals, transpose, compact, Dirac, HD
export sumrule, chainrule, conditioningrule, bayesrule
export filtr, filtersmoother, estep

#############################################


end