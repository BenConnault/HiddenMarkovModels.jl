module HiddenMarkovModels

# For using   
using StatsFuns: normcdf 
using Distributions: wsample, sample

# For extending   
using  StatsBase
import StatsBase: loglikelihood 
import Base: rand
# using Base: rand, norm, length



include("utils/filtering-utils.jl")
include("utils/rkhs.jl")


### Legacy "Dynamic Discrete Model" back-end

	include("ddm/dynamicdiscretemodel.jl")  # `DynamicDiscreteModel`
	include("ddm/simulate.jl")              # `rand`
	include("ddm/loglikelihood.jl")         # forward filtering + loglikelihood + jacobian
	include("ddm/estep.jl")                 # e-step of the EM algorithm.
	include("ddm/emalgorithm.jl")           # numerical optimiation for the M-step.
	include("ddm/viterbi.jl")
	include("ddm/filtering.jl")				#plain computation of the filter-smoother. Not used in loglikelihood or estep but sometimes useful.	
	include("ddm/toymodel.jl")				#Useful for testing and examples.	
	include("ddm/dhmm.jl")					# a thin layer on top of the "Dynamic Discrete Model" back-end, all previous functions

### Core methods

	include("main/abstract-hidden-markov.jl")
	include("main/generic-filter.jl")
	include("main/discrete.jl")
	include("main/strict-hidden-markov-kernel-methods.jl")  
	include("main/strict-hidden-markov-low-rank-kernel-methods.jl")  
	include("main/linear-gaussian.jl")  #inherits from StrictHMM
	include("main/strict-hidden-markov-particle-methods.jl")  

	export HiddenMarkovModels, LinearGaussianHMM, DiscreteHMM    	    # model types
	export KKF, PF														    # filtering techniques
	export filtr, loglikelihood, filter_smoother, eweights!             # methods
	export viterbi

include("utils/stochasticmatrices.jl")
# include("utils/tensors.jl")
# include("utils/distances.jl")


end

