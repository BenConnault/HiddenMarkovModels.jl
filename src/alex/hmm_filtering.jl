### Forward-Backward Algorithm
function forward_backward(hmm::HMM,o::Vector; scaling=true)
	if scaling
		alpha, log_p_obs, coeff = forward(hmm,o; scaling=true)
		beta = backward(hmm,o; scale_coeff=coeff)
	else
		alpha, p_obs = forward(hmm,o; scaling=false)
		log_p_obs = log(p_obs)
	    beta = backward(hmm,o)
	end
	return alpha, beta, log_p_obs
end

function forward(hmm::HMM, o::Vector; scaling=true)
	n_obs = length(o)

	# alpha[t,i] = probability of being in state 'i' given o[1:t]
	alpha = zeros(n_obs, hmm.n) 

	# base case (initialize at start)
	for i = 1:hmm.n
		alpha[1,i] = hmm.p[i] * pdf(hmm.B[i],o[1])
	end

	if scaling
		c = (Float64)[] # scaling coefficients
		push!(c,1./sum(alpha[1,:]))
		alpha[1,:] *= c[end] 
	end

	# induction step
	for t = 2:n_obs
		for j = 1:hmm.n
			for i = 1:hmm.n
				alpha[t,j] += hmm.A[i,j] * alpha[t-1,i]
			end
			alpha[t,j] *= pdf(hmm.B[j],o[t])
		end
		if scaling
			push!(c,1./sum(alpha[t,:]))
			alpha[t,:] *= c[end]
		end
	end

	# Calculate likelihood (or log-likelihood) of observed sequence
	if scaling
		log_p_obs = -sum(log(c)) # see Rabiner (1989), eqn 103
		return (alpha,log_p_obs,c)
	else
		p_obs = sum(alpha[end,:]) 
		return (alpha,p_obs)
	end
end

function backward(hmm::HMM, o::Vector; scale_coeff=nothing)
	# scale_coeff are 1/sum(alpha[t,:]) calculated by forward algorithm
	n_obs = length(o)

	# beta[t,i] = probability of being in state 'i' and then obseverving o[t+1:end]
	beta = zeros(n_obs, hmm.n)

	# base case (initialize at end)
	if scale_coeff == nothing
		beta[end,:] += 1
	else
		if length(scale_coeff) != n_obs
			error("scale_coeff is improperly defined (wrong length)")
		end
		beta[end,:] += scale_coeff[end]
	end

	# induction step
	for t = reverse(1:n_obs-1)
		for i = 1:hmm.n
			for j = 1:hmm.n
				beta[t,i] += hmm.A[i,j] * pdf(hmm.B[j],o[t+1]) * beta[t+1,j]
			end
		end
		if scale_coeff != nothing
			beta[t,:] *= scale_coeff[t]
		end
	end

	return beta
end

### Viterbi

function viterbi(hmm::HMM, o::Vector)
	n_obs = length(o)

	# delta[i,j] = highest probability of state sequence ending in state j on step i
	# psi[i,j] = most likely state on step i-1 given state j on step i (argmax of deltas)
	delta = zeros(n_obs, hmm.n)
	psi = ones(Int, n_obs, hmm.n)

	# base case, psi[:,1] is ignored so don't initialize
	for i = 1:hmm.n
		delta[1,i] = hmm.p[i] .* pdf(hmm.B[i],o[1])
	end

	# induction step
	for t = 2:n_obs
		for j = 1:hmm.n
			delta[t,j],psi[t,j] = findmax(hmm.A[:,j].*delta[t-1,:]')
			delta[t,j] *= pdf(hmm.B[j],o[t])
		end
	end

	# backtrack to uncover the most likely path / state sequence
	q = zeros(Int,n_obs) # vector holding state sequence
	q[end] = indmax(delta[end,:])

	# backtrack recursively
	for t = reverse(1:n_obs-1)
		q[t] = psi[t+1,q[t+1]]
	end
	return q
end

### Smoothed state sequence based on forward-backward algorithm
function smoothed_forward_backward(hmm::HMM, o::Vector; scaling=true)
	log_p_obs,alpha,beta,x,g = calc_stats(hmm,o;scaling=scaling)
	return g,log_p_obs
end
