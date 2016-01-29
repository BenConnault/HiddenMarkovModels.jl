module HiddenMarkovModels

importall DynamicDiscreteModels

# When I get around to spinning of a Markov.jl
# import Markov: rsm, nsm, z2q, q2z

export 	calibrate!, simulate, loglikelihood, mle, dim, 
		em, viterbi,
		baumwelch, hmm, theta2ab


#source files
include("hiddenmarkovmodel.jl")
include("stochasticmatrices.jl")

end