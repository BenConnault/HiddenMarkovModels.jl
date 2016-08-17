function filtr(model::DynamicDiscreteModel,data::Array{Int,1})
	T=length(data)
	dx,dy=size(model.mu)

	filter=Array(Float64,dx,T)
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

	smoother=Array(Float64,dx,T)
	smoother[:,T]=1
	for t=T-1:-1:1
		rho=0.0
		for ix=1:dx
			smoother[ix,t]=0
			for jx=1:dx
				smoother[ix,t]+=model.m[ix,data[t],jx,data[t+1]]*smoother[jx,t+1]
			end
			rho+=smoother[ix,t]
		end
		for ix=1:dx
			smoother[ix,t]/=rho
		end
	end

	conditional=Array(Float64,dx,dx,T)
	for t=1:T-1
		tempsum=0.0
		for jx=1:dx
			for ix=1:dx
				conditional[ix,jx,t]=model.m[ix,data[t],jx,data[t+1]]
				conditional[ix,jx,t]*=filter[ix,t]
				conditional[ix,jx,t]*=smoother[jx,t+1]
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
	filter,smoother,conditional
end