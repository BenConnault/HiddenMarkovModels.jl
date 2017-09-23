module HiddenMarkovModels

# For using   
using StatsFuns: normcdf 
using Distributions: wsample

# For extending   
using  StatsBase
import StatsBase: loglikelihood   # loglikelihood
import Base: rand
# using Base: rand, norm, length


include("utils/stochasticmatrices.jl")
### cheap export: copy-paste in your workspace the line below if you want to use the internal methods
# using HiddenMarkovModels: rsm   

include("utils/tensors.jl")
# using HiddenMarkovModels: ei, vecpq, opnorm, partialtrace, rortho

include("utils/distances.jl")
# using HiddenMarkovModels: dhilbert, dtv

include("utils/filtering-utils.jl")
include("utils/rkhs.jl")

# include("utils/kde.jl")

export coef!, loglikelihood

### "Dynamic Discrete Model" back-end

	include("ddm/dynamicdiscretemodel.jl")  # `DynamicDiscreteModel`
	include("ddm/simulate.jl")              # `rand`
	include("ddm/loglikelihood.jl")         # forward filtering + loglikelihood + jacobian
	include("ddm/estep.jl")                 # e-step of the EM algorithm.
	include("ddm/emalgorithm.jl")           # numerical optimiation for the M-step.
	include("ddm/viterbi.jl")
	include("ddm/filtering.jl")				#plain computation of the filter-smoother. Not used in loglikelihood or estep but sometimes useful.	
	include("ddm/toymodel.jl")				#Useful for testing and examples.	
	include("ddm/dhmm.jl")					# a thin layer on top of the "Dynamic Discrete Model" back-end, all previous functions

	export 	em, viterbi, filtr, baumwelch, hmm, theta2ab

### Models

	include("models/abstract-hidden-markov.jl")
	include("models/strict-hidden-markov.jl")   #including discrete

	# using Kalman
	include("models/linear-gaussian.jl")
	# include("models/discrete.jl")


	# export lgmodel, dhmm

### Filtering

	include("filtering/generic-filter.jl")
	include("filtering/kalman-filter.jl")
	include("filtering/particle-filter.jl")
	# include("filtering/high-dim-kernel-filter.jl")   #TO ADAPT
	# include("filtering/kernel-density-filter.jl")    #TO ADAPT
	
	export StrictHiddenMarkov
	export KKF, PF
	# , BF, LRKF, LRBF
	export filtr



### Commented out

	# importall JuMP, Ipopt
	# include("utils/wasserstein.jl")   # needs external optimization package
    # using HiddenMarkovModels: wasserstein, dwasserstein

end

