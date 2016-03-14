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



function filtr3{H1,H2,Hx<:RKHS,Hy<:RKHS}(transition1::RKHSMap{H1,Hy},transition2::RKHSMap{H2,Hx},initial,data;lambda=0.0)
	@assert H1 <: Union{Hx,RKHS2{Hx,Hy}}
	# `transition1` is P(Y_{t+1}|X_t,Y_t) expressed in (Bx1 x By1) -> By2  
	# [FOR NOW] transition2 is P(X_{t+1}|X_t,Y_{t+1})  (valid IF P(X_{t+1}|X_t,Y_t,Y_{t+1})=P(X_{t+1}|X_t,Y_{t+1}) )
	#   expressed in (Bx1 x By2) -> Bx2 
	# [IN THE FUTURE] transition2 is P(X_{t+1}|X_t,Y_t,Y_{t+1})
	# `initial` is P(X_1 | Y_1=y_1)
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

		### MK1

		delta_yt=project(data[t],By1)
		
		#express (P(X_t|Y_{1:t}=y_{1:t}) x deta_y_t) in Bx1 x By2
		prior=kron(delta_yt,filter[t])

		# marginal x conditional = joint (X_t x Y_t x Y_{t+1})
		joint=reshape(scale(prior,mk1),dx,dy,dy)

		#joint after marginalizing y_t which has been observed. (X_t x Y_{t+1})
		#there is probably a faster way of doing expansion then reduction by doing 
		#direct contraction
		joint=squeeze(sum(joint,2),2)

		#posterior X_t | Y_{t+1}
		posterior=joint./sum(joint,1)

		# (X_t | y_{t+1}) = (X_t | Y_{t+1}) x (Y_t+1 = y_t+1)
		delta_yt1=project(data[t+1],By2)
		posterior=posterior*delta_yt1


		### MK2
		#mk2=P(X_{t+1}|X_t,Y_{t+1})

		filter_Bx2=At_mul_B(kron(delta_yt1,posterior),mk2)
		filter[t+1]=project(filter_Bx2,Bx1)

	end
	filter
end


function filtersmoother{H,Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{H,RKHS2{Hx,Hy}},initial,data;lambda=0.0)
	@assert H <: Union{Hx,RKHS2{Hx,Hy}}
	filter=filtr(transition,initial,data,lambda=lambda)

	T=length(data)

	smoother=Dict(T => filter[T])

	for t=T-1:-1:1
		# P(X_t | Y_{1:t}=y_{1:t})
		prior=filter[t]											

		# P(X_{t+1},Y_{t+1} | X_t, Y_t=y_t) = P(X_{t+1},Y_{t+1} | X_t, Y_{1:t}=y_{1:t})
		conditional=sumrule(Dirac(Hy,data[t]),transition)		

		# P(X_t | X_{t+1},Y_{t+1}, Y_{1:t}=y_{1:t})
		posterior=bayesrule(prior,conditional,lambda=lambda)	

		# P(X_t | X_{t+1}, Y_{1:t+1}=y_{1:t+1}) = P(X_t | X_{t+1}, Y_{1:T}=y_{1:T})
		revdict=sumrule(Dirac(Hy,data[t+1]),posterior) 			
		
		# P(X_t | Y_{1:T}=y_{1:T})
		smoother[t]=sumrule(smoother[t+1],revdict)              
	end
	filter,smoother
end

function estep{H,Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{H,RKHS2{Hx,Hy}},initial,data;lambda=0.0)
	@assert H <: Union{Hx,RKHS2{Hx,Hy}}
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

		# P(X_t | Y_{1:t}=y_{1:t})
		prior=filter[t]											

		# P(X_{t+1},Y_{t+1} | X_t, Y_t=y_t) = P(X_{t+1},Y_{t+1} | X_t, Y_{1:t}=y_{1:t})
		conditional=sumrule(Dirac(Hy,data[t]),transition)		

		# P(X_t | X_{t+1},Y_{t+1}, Y_{1:t}=y_{1:t})
		posterior=bayesrule(prior,conditional,lambda=lambda)	

		# P(X_t | X_{t+1}, Y_{1:t+1}=y_{1:t+1}) = P(X_t | X_{t+1}, Y_{1:T}=y_{1:T})
		revdict=sumrule(Dirac(Hy,data[t+1]),posterior) 			

		# P(X_t,X_{t+1} | Y_{1:T}=y_{1:T})			
		eweights[t]=transpose(chainrule(smoother,revdict)) 		

		# P(X_t | Y_{1:T}=y_{1:T})
		smoother=sumrule(smoother,revdict)              		
	end
	eweights
end

