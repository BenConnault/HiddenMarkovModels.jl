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

using JuMP, Ipopt

import Base.length

line(x)=reshape(x,1,lengh(x))

include("rkhs_types.jl")
include("rkhs_vptree.jl")
# include("rkhs_filtering.jl")

export GaussianRKHS, DiscreteRKHS, Point, RKHSVector, RKHSMap, KernelDistance
export VPTree, knn
export dimension,length

#############################################


end