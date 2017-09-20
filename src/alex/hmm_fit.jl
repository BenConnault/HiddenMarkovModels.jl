
# TODO: fit(D::Distribution, x::Vector{Float}, w::Vector{Float}) --> weighted maximum likelihood estimation...

# Fitting function, add more methods here
function fit!(hmm::HMM, o; method=:baum_welch, max_iter=100, tol=1e-5, scaling=true)
	if method == :baum_welch
		return baum_welch!(hmm, o, max_iter, tol, scaling)
	end
end

function baum_welch!(hmm::HMM, o, max_iter, tol, scaling)
	# Fit hmm parameters given set of observation sequences
	# Setting tol to NaN will prevent early stopping, resulting in 'max_iter' iterations
	#n_seq = length(sequences)

    # convergence history of the fit, log-liklihood
    ch = (Float64)[]

    for k = 1:max_iter
    	push!(ch,0.0)

		# E step
		log_p_obs,alpha,beta,x,g = calc_stats(hmm,o,scaling)
		ch[end] += log_p_obs #(log_p_obs + log(length(o)))

		# M step (re-estimation)
		re_estimate!(hmm,o,x,g)
		
		if length(ch)>1 && (ch[end]-ch[end-1] < tol)
			println("Baum-Welch converged, stopping early")
			break
		end
	end

	return ch
end

function re_estimate!(hmm::HMM,o,x,g)
	# Update state transition matrix
    for i = 1:hmm.n
        denom = sum(g[1:end-1,i]) # Expected number of transitions from state 'i'
        for j = 1:hmm.n
            # Numerator is the expected number of transitions from state 'i' to 'j' 
            hmm.A[i,j] = sum(x[:,i,j]) / denom
        end
    end

	# Update emission probability distributions
	for i = 1:hmm.n
		# weight each observation by probability of being in state i
		hmm.B[i] = fit_mle(typeof(hmm.B[i]), o, g[:,i])
	end

	# Re-estimate hmm.p (initial state probabilities)
	hmm.p = vec(g[1,:])
end

function calc_stats(hmm::HMM,o,scaling)
	## Single E-M iteration in the Baum-Welch procedure
	n_obs = length(o)

	# Calculate forward/backward probabilities
	alpha,beta,log_p_obs = forward_backward(hmm,o;scaling=scaling)

	# x[t,i,j] = probability of being in state 'i' at 't' and then in state 'j' at 't+1'
	x = zeros(n_obs-1, hmm.n, hmm.n)
	for t = 1:(n_obs-1)
		for i = 1:hmm.n
			for j = 1:hmm.n
				x[t,i,j] = alpha[t,i] * hmm.A[i,j] * pdf(hmm.B[j],o[t+1]) * beta[t+1,j]
			end
		end
		x[t,:,:] ./= sum(x[t,:,:]) # normalize to achieve probabilities
	end

	# g[t,i] = probability of being in state 'i' at step 't' given all observations
	g = alpha .* beta
	g ./= sum(g,2)   # normalize across states

	return log_p_obs, alpha, beta, x, g
end