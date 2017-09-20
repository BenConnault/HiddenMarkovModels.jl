module HiddenMarkovModels

importall Distributions

importall JuMP, Ipopt
# used for
#  - Wasserstein distance
#  - 


using StatsBase: sample, StatisticalModel
using StatsFuns: normcdf
using Distributions: wsample


import Base: rand, norm, length



######### Alex

export	HMM, forward_backward, viterbi, smoothed_forward_backward, fit!

include("alex/hmm_types.jl")
include("alex/hmm_filtering.jl")
include("alex/hmm_fit.jl")

######## Ben

include("utils/stochasticmatrices.jl")
export rsm

include("utils/tensors.jl")
export ei, vecpq, opnorm, partialtrace, rortho

include("utils/distances.jl")
export wasserstein, dwasserstein, dhilbert, dtv

include("utils/filtering-utils.jl")
include("utils/rkhs.jl")

# include("utils/kde.jl")


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

	include("dhmm/dhmm.jl")

	export 	coef!, rand, loglikelihood, mle, dim, 
		em, viterbi, filtr,
		baumwelch, hmm, theta2ab


### Experimental RKHS filtering

	# include("rkhs/vptree.jl")
	# include("rkhs/tupletype.jl")
	# include("rkhs/types.jl")
	# include("rkhs/project.jl")
	# include("rkhs/filtering.jl")

	# export instantiate
	# export VPTree, knn, Distance, evaluate
	# export AtomicRKHS, RKHS, GaussianRKHS, DiscreteRKHS, LaplaceRKHS, GuilbartRKHS, RKHSBasis, RKHSVector, RKHSMap, KernelDistance, RKHSBasisTree, rkhs, kernel, gramian
	# export dimension,length
	# export FilteringAlgorithm, Strict, General, Alt, project, filtr, filtr_smoothr 

	# line(x)=reshape(x,1,lengh(x))

### Models

	using Kalman
	include("models/abstract-hidden-markov.jl")
	include("models/strict-hidden-markov.jl")
	include("models/linear-gaussian.jl")
	include("models/discrete.jl")


	export lgmodel, dhmm

### Kernel Filtering

	include("filtering/generic-filter.jl")
	include("filtering/kernel-filter.jl")
	include("filtering/high-dim-kernel-filter.jl")
	include("filtering/particle-filter.jl")
	include("filtering/basis-filter.jl")
	# include("kernelfiltering/altfiltering.jl")

	export StrictHiddenMarkovModel
	export KF, PF, BF, LRKF, LRBF
	export filtr

end



### Discrete RKHS for playing around

# include("drkhs/DRKHS.jl")