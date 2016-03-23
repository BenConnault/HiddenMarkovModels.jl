# """
# Filtering in a hidden Markov model.
# 	filter(transition,initial,data)
# """

function filtr{H,Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{H,RKHS2{Hx,Hy}},initial,data;lambda=0.0)
	@assert H <: Union{Hx,RKHS2{Hx,Hy}}
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

		predict=transpose(compact(RKHSMap(predict)))

		filter[t+1]=sumrule(Dirac(Hy,data[t+1]),conditioningrule(predict,lambda=lambda))
	end
	filter
end

using Plots

gramx{Hx,Hy}(points::AbstractVector,transition::RKHSMap{Hx,Hy})=kernel(Hx,points,transition.leftpoints)
gramx{Hx,Hy1,Hy2}(points::AbstractVector,transition::RKHSMap{RKHS2{Hx,Hy1},Hy2})=
	kernel(Hx,points,line(unpack(transition.leftpoints)[1]))

function filtr2{H1,H2,Hx<:RKHS,Hy<:RKHS}(transition1::RKHSMap{H1,Hy},transition2::RKHSMap{H2,Hx},initial,data;lambda=0.0)
	@assert H1 <: Union{Hx,RKHS2{Hx,Hy}}
	# transition1 is P(Y_{t+1}|X_t,Y_t)
	# [FOR NOW] transition2 is P(X_{t+1}|X_t,Y_{t+1})  (valid IF P(X_{t+1}|X_t,Y_t,Y_{t+1})=P(X_{t+1}|X_t,Y_{t+1}) )
	# [IN THE FUTURE] transition2 is P(X_{t+1}|X_t,Y_t,Y_{t+1})

	T=length(data)
	filter=Dict(1=>initial)

	#for diagnostic
	# matrices=Dict(1=>zeros(1,1))
	
	# Gram matrix from support points of predict (rightpoints of transition2)
	# to x-support points of transition 1 
	Gx=gramx(transition2.rightpoints,transition1)   
	
	Gy=kernel(Hy,transition1.rightpoints,line(transition1.rightpoints))
	# Gyi=inv(Symmetric(Gy))
	# Gyi=(Gyi.*(abs(Gyi).>.01))

	for t=1:T-1
		prior=filter[t]
		conditional=sumrule(Dirac(Hy,data[t]),transition1)
		predict=bayesrule(prior,conditional,Gx,Gy,lambda=lambda)
		
		# predict,matrices[t+1]=predict
		# yy=linspace(-2,2,20)
		# mu(y)=moment(sumrule(Dirac(Hy,y),predict),1)
		# plot(yy,map(mu,yy))

		predict=sumrule(Dirac(Hy,data[t+1]),predict)
		transit=sumrule(Dirac(Hy,data[t+1]),transition2)
		filter[t+1]=sumrule(predict,transit)
	end
	filter
end
