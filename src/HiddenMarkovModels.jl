module HiddenMarkovModels

import Base.rand
import Base.norm
import Base.length

using StatsBase:sample,WeightVec,StatisticalModel
import Distributions: wsample, Dirichlet
using JuMP, Ipopt


######### Alex
importall Distributions

export	HMM, forward_backward, viterbi, smoothed_forward_backward, fit!

include("hmm_types.jl")
include("hmm_filtering.jl")
include("hmm_fit.jl")

######## Ben

# rsm, nsm, z2q, q2z
include("utils_stochasticmatrices.jl")


### "Dynamic Discrete Model" back-end

	# Define a StatisticalModel interface and provide convenience functions such as numerical optimization of the likelihood based on Optim.jl.
	# This is a good candidate for sending upstream to eg. StatsBase.
	include("ddm/statisticalmodels-stopgap.jl")

	include("ddm/dynamicdiscretemodel.jl")  # `abstract DynamicDiscreteModel <: StatisticalModel` interface
	include("ddm/simulate.jl")              # `rand`
	include("ddm/loglikelihood.jl")         # forward filtering + loglikelihood + jacobian
	include("ddm/estep.jl")                 # e-step of the EM algorithm.
	include("ddm/emalgorithm.jl")           # numerical optimiation for the M-step.
	include("ddm/viterbi.jl")
	include("ddm/filtering.jl")				#plain computation of the filter-smoother. Not used in loglikelihood or estep but sometimes useful.	
	include("ddm/toymodel.jl")				#Useful for testing and examples.	

### "Discrete Hidden Markov Model": a thin layer on top of the "Dynamic Discrete Model" back-end

	include("dhmm.jl")

	export 	coef!, rand, loglikelihood, mle, dim, 
		em, viterbi, filtr,
		baumwelch, hmm, theta2ab


### Experimental RKHS filtering

	include("rkhs/vptree.jl")
	include("rkhs/tupletype.jl")
	include("rkhs/types.jl")
	include("rkhs/project.jl")
	include("rkhs/filtering.jl")





# line(x)=reshape(x,1,lengh(x))


export instantiate
export VPTree, knn, Distance, evaluate
export AtomicRKHS, RKHS, GaussianRKHS, DiscreteRKHS, RKHSBasis, RKHSVector, RKHSMap, KernelDistance, RKHSBasisTree, rkhs, kernel, gramian
export dimension,length
export project,filtr



end