## HMM.jl, contains constructors for HMM objects and 'generate()' function

# HMM -- HMM with emission probability distribution 'C'
type HMM{C<:Distribution}   <: StatisticalModel
	n::Int             # Number of hidden states
	A::Matrix{Float64} # Estimated state-transition matrix A[i,j] = Pr[i->j]
	B::Vector{C}       # Estimated emission probability distributions
	p::Vector{Float64} # Estimiated initial state probabilities

	# Notes:
	#   "A" is a NxN matrix, rows sum to one

	# To do:
	#    Allow B to depend on other observables, for observation o and param k, B(o|k)
end


# TO DO: add a constructor each B[i] is drawn from some distribution specified by hyperparams

function HMM(n::Int,C::Distribution)
	# Randomize state-transition matrix
	A = rand(n,n)
	A ./= sum(A,2) # normalize rows	
	
	# Specify a distribution of type C for each state
	B = (Distribution)[]
	for i = 1:n
		push!(B,deepcopy(C))
	end

	# Randomize initial state probabilities
	p = rand(n)
	p ./= sum(p)

	return HMM(n,A,B,p)
end

function HMM(A::Matrix{Float64},C::Distribution)
	# determine number of states
	@assert size(A,1) == size(A,2)
	n = size(A,1)

	# Specify a distribution of type C for each state
	B = (Distribution)[]
	for i = 1:n
		push!(B,deepcopy(C))
	end

	# Randomize initial state probabilities
	p = rand(n)
	p ./= sum(p)

	return HMM(n,A,B,p)
end

function HMM(A::Matrix{Float64},Bmat::Matrix{Float64})
	# Initialize a discrete HMM, with Categorical() emissions
	n = size(A,1)

	# Each row of Bmat specifies a Categorical pdf
	B = (Distribution)[]
	for i = 1:n
		push!(B,Categorical(vec(Bmat[i,:])))
	end
	
	return HMM(A,B)
end

function HMM(A::Matrix{Float64},B::Vector{Distribution})
	# determine number of states
	@assert size(A,1) == size(A,2)
	n = size(A,1)
	@assert length(B) == n

	# Randomize initial state probabilities
	p = rand(n)
	p ./= sum(p)

	return HMM(n,A,B,p)
end

function HMM(A::Matrix{Float64},B,p::Vector{Float64})
	@assert sum(p)==1
	hmm = HMM(A,B)
	hmm.p = p # reset initial probabilities
	return hmm
end

function rand(hmm::HMM, n_obs::Int)
	# Generate a sequence of n_obs observations from an HMM.

	# Sequence of states and observations
	s = zeros(Int,n_obs) # states
	o = zeros(n_obs)     # observations

	# Choose initial state with probabilities weighted by "init_state"
	s[1] = rand(Categorical(hmm.p))  # hmm.p are the initial state probabilities
	o[1] = rand(hmm.B[s[1]])         # draw observation given initial state

	# Construct categorical distributions from each row of A
	Ac = (Categorical)[]
	for i = 1:hmm.n
		push!(Ac,Categorical(vec(hmm.A[i,:])))
	end

	# Iterate drawing observations and updating state
	for t = 2:n_obs
		s[t] = rand(Ac[s[t-1]])   # sample from appropriate row of state-transition matrix
		o[t] = rand(hmm.B[s[t]])  # sample from emission probability distribution
	end

	# return sequence of states and observations
	return (s,o)
end
