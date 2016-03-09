# """
# Filtering in a hidden Markov model.
# 	filter(transition,initial,data)
# """

function filtr{H,Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{H,RKHS2{Hx,Hy}},initial,data;lambda=0.0)
	H<:Union{Hx,RKHS2{Hx,Hy}}
	T=length(data)
	updates=Dict(1=>initial)
	predicts=Dict(2=>sumrule(RKHSLeftElement(chainrule(updates[1],Dirac(Hy,data[1]))),transition))
	updates[2]=sumrule(Dirac(Hy,data[2]),conditioningrule(transpose(compact(RKHSMap(predicts[2]))),lambda=lambda))

	for t=2:T-1
		
		# If multiple constructor calls are expensive,
		# I could create deltay before the loop and use a setter to modify inner fields
		deltay=Dirac(Hy,data[t])    

		# (1) create product measure ([yesterday's predict] x [data Dirac]) [chainrule]
		# + RKHSLeftElement for casting as vector in H1xH2 rather than a (H1,H2) matrix
		# (2) transition it to t+1 [sumrule] 
		predicts[t+1]=sumrule(RKHSLeftElement(chainrule(updates[t],Dirac(Hy,data[t]))),transition)
		# An alternative would be:
		# sumrule(updates[t],(sumrule(deltay,transition)))
		# (1) partial transition of [data Dirac]
		# (2) transition of [yesterday's predict]
		updates[t+1]=sumrule(Dirac(Hy,data[t+1]),conditioningrule(transpose(compact(RKHSMap(predicts[t+1]))),lambda=lambda))
	end
	predicts,updates
end

function filtersmoother{H,Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{H,RKHS2{Hx,Hy}},initial,data;lambda=0.0)
	predicts,updates=filtr(transition,initial,data,lambda=lambda)

	T=length(data)

	posteriors=Dict(T => updates[T])
	prior=updates[T-1]
	conditional=sumrule(Dirac(Hy,data[T-1]),transition)
	posterior=bayesrule(prior,conditional,lambda=lambda)
	revdicts=Dict(T-1 => sumrule(Dirac(Hy,data[T]),posterior))
	posteriors[T-1]=sumrule(posteriors[T],revdicts[T-1])

	for t=T-1:-1:1
		prior=updates[t]
		conditional=sumrule(Dirac(Hy,data[t]),transition)
		posterior=bayesrule(prior,conditional,lambda=lambda)
		revdicts[t]=sumrule(Dirac(Hy,data[t+1]),posterior)
		posteriors[t]=sumrule(posteriors[t+1],revdicts[t])
	end
	# predicts,updates,revdicts,posteriors    #could return predicts and updates if needed but I don't know a used case as of now
	revdicts,posteriors
end

function estep{H,Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{H,RKHS2{Hx,Hy}},initial,data;lambda=0.0)

end

# function estep(transition,initial,data)
# 	revdicts,posteriors=filtersmoother(transition,initial,data)

# 	eweights=
# end


