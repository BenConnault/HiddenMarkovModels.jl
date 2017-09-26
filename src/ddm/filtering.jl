function filtr(model::DynamicDiscreteModel,data::Array{Int,1})
	T=length(data)
	dx,dy=size(model.mu)

	filter=Array{Float64}(dx,T)
	filter[:,1]=model.mu[:,data[1]]/sum(model.mu[:,data[1]])
	for t=2:T
		rho=0.0
		for jx=1:dx
			filter[jx,t]=0
			for ix=1:dx
				filter[jx,t]+=model.m[ix,data[t-1],jx,data[t]]*filter[ix,t-1]
			end
			rho+=filter[jx,t]
		end
		for jx=1:dx
			filter[jx,t]/=rho
		end
	end

	# backward "smoothing" via the 2-filter formula.
	# `backward` stores p(y_{t+1:T}|y_t,x_t) and NOT the smoother p(x_t|y_{1:T}) 
	# however the smoother can be read off as the marginal of conditional below
	backward=Array{Float64}(dx,T)
	backward[:,T]=1
	for t=T-1:-1:1
		rho=0.0
		for ix=1:dx
			backward[ix,t]=0
			for jx=1:dx
				backward[ix,t]+=model.m[ix,data[t],jx,data[t+1]]*backward[jx,t+1]
			end
			rho+=backward[ix,t]
		end
		for ix=1:dx
			backward[ix,t]/=rho
		end
	end


	# conditional[:,:,t] stores p(x_{t+1},x_t| y_{1:T})
	# conditional[:,:,T] is uninitialized -- garbage
	conditional=Array{Float64}(dx,dx,T)
	for t=1:T-1
		tempsum=0.0
		for jx=1:dx
			for ix=1:dx
				conditional[ix,jx,t]=model.m[ix,data[t],jx,data[t+1]]
				conditional[ix,jx,t]*=filter[ix,t]
				conditional[ix,jx,t]*=backward[jx,t+1]
				tempsum+=conditional[ix,jx,t]
			end
		end
		for jx=1:dx
			for ix=1:dx
				#note that there is no risk of division by zero 
				#reason: this is called only on _observed_ data which must thus have non-zero probability
				conditional[ix,jx,t]=conditional[ix,jx,t]/tempsum
			end
		end
	end
	filter,backward,conditional
end