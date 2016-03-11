module HiddenMarkovModels

importall DynamicDiscreteModels
importall Distributions
using StatsBase:sample,WeightVec,StatisticalModel
import Base.norm


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
import Distributions.moment

include("rkhs_types.jl")
include("rkhs_calculus.jl")
include("rkhs_filtering.jl")
include("rkhs_spaces.jl")

export line
export RKHSLeftElement, RKHSRightElement, RKHSMap, RKHS2, marginal, marginals, transpose, compact, distance, Dirac, HD, HG, moment
export sumrule, chainrule, conditioningrule, bayesrule
export filtr, filtr2, filtersmoother, estep

#############################################


end