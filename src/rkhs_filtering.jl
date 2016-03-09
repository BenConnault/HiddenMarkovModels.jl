# """
# Filtering in a hidden Markov model.
# 	filter(transition,initial,data)
# """

function filtr{H,Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{H,RKHS2{Hx,Hy}},initial,data;lambda=0.0)
	H<:Union{Hx,RKHS2{Hx,Hy}}
	T=length(data)
	filter=Dict(1=>initial)

	for t=1:T-1
		
		# If multiple constructor calls are expensive,
		# I could create deltay before the loop and use a setter to modify inner fields
		deltay=Dirac(Hy,data[t])    

		# (1) create product measure ([yesterday's predict] x [data Dirac]) [chainrule]
		# + RKHSLeftElement for casting as vector in H1xH2 rather than a (H1,H2) matrix
		# (2) transition it to t+1 [sumrule] 
		predict=sumrule(RKHSLeftElement(chainrule(filter[t],Dirac(Hy,data[t]))),transition)
		# An alternative would be:
		# sumrule(filter[t],(sumrule(deltay,transition)))
		# (1) partial transition of [data Dirac]
		# (2) transition of [yesterday's predict]
		filter[t+1]=sumrule(Dirac(Hy,data[t+1]),conditioningrule(transpose(compact(RKHSMap(predict))),lambda=lambda))
	end
	filter
end

function filtersmoother{H,Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{H,RKHS2{Hx,Hy}},initial,data;lambda=0.0)
	filter=filtr(transition,initial,data,lambda=lambda)

	T=length(data)

	smoother=Dict(T => filter[T])

	for t=T-1:-1:1
		prior=filter[t]											# P(X_t | Y_{1:t}=y_{1:t})
		conditional=sumrule(Dirac(Hy,data[t]),transition)		# P(X_{t+1},Y_{t+1} | X_t, Y_t=y_t) = P(X_{t+1},Y_{t+1} | X_t, Y_{1:t}=y_{1:t})
		posterior=bayesrule(prior,conditional,lambda=lambda)	# P(X_t | X_{t+1},Y_{t+1}, Y_{1:t}=y_{1:t})
		revdict=sumrule(Dirac(Hy,data[t+1]),posterior) 			# P(X_t | X_{t+1}, Y_{1:t+1}=y_{1:t+1}) = P(X_t | X_{t+1}, Y_{1:T}=y_{1:T})
		smoother[t]=sumrule(smoother[t+1],revdict)              # P(X_t | Y_{1:T}=y_{1:T})
	end
	filter,smoother
end

function estep{H,Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{H,RKHS2{Hx,Hy}},initial,data;lambda=0.0)
	filter=filtr(transition,initial,data,lambda=lambda)

	T=length(data)

	smoother=filter[T]
	prior=filter[T-1]											
	conditional=sumrule(Dirac(Hy,data[T-1]),transition)		
	posterior=bayesrule(prior,conditional,lambda=lambda)	
	revdict=sumrule(Dirac(Hy,data[T-1+1]),posterior)
	eweights=Dict(T-1 => transpose(chainrule(smoother,revdict))) 			
	smoother=sumrule(smoother,revdict)              

	for t=T-2:-1:1
		prior=filter[t]											# P(X_t | Y_{1:t}=y_{1:t})
		conditional=sumrule(Dirac(Hy,data[t]),transition)		# P(X_{t+1},Y_{t+1} | X_t, Y_t=y_t) = P(X_{t+1},Y_{t+1} | X_t, Y_{1:t}=y_{1:t})
		posterior=bayesrule(prior,conditional,lambda=lambda)	# P(X_t | X_{t+1},Y_{t+1}, Y_{1:t}=y_{1:t})
		revdict=sumrule(Dirac(Hy,data[t+1]),posterior) 			# P(X_t | X_{t+1}, Y_{1:t+1}=y_{1:t+1}) = P(X_t | X_{t+1}, Y_{1:T}=y_{1:T})
		eweights[t]=transpose(chainrule(smoother,revdict)) 		# P(X_t,X_{t+1} | Y_{1:T}=y_{1:T})			
		smoother=sumrule(smoother,revdict)              		# P(X_t | Y_{1:T}=y_{1:T})
	end
	eweights
end

# function estep(transition,initial,data)
# 	revdicts,posteriors=filtersmoother(transition,initial,data)

# 	eweights=
# end


