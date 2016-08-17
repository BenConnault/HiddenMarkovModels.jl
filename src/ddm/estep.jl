"""

E-step of the EM-algorithm.

	filtersmoother(model,data,w)

Calibrates a w[x,y,x',y'] array of weights such that the EM-step is obtained by maximizing 
sum_{x,y,x',y'} w[x,y,x',y']\\*log(m[x,y,x',y']) which in Julia is simply `sum(w.*m)`.
The weights are obtained by running the filter/smoother (aka forward/backward) algorithm.
The `EMalgorithm` function wraps `filtersmoother` along with a generic numerical optimization
step, but more efficient (even closed-form) M-step optimization may be available in particular
models, such as Hidden Markov Models: if this the case call `filtersmoother` to build the weights and 
implement your own optimization routine.

"""
function estep(model::DynamicDiscreteModel,data::Array{Int,1},w::Array{Float64,4})
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

	conditional=Array(Float64,dx,dx)
	for t=1:T-1
		tempsum=0.0
		for jx=1:dx
			for ix=1:dx
				conditional[ix,jx]=model.m[ix,data[t],jx,data[t+1]]
				conditional[ix,jx]*=filter[ix,t]
				conditional[ix,jx]*=smoother[jx,t+1]
				tempsum+=conditional[ix,jx]
			end
		end
		for jx=1:dx
			for ix=1:dx
				#note that there is no risk of division by zero 
				#reason: this is called only on _observed_ data which must thus have non-zero probability
				w[ix,data[t],jx,data[t+1]]+=conditional[ix,jx]/tempsum
			end
		end
	end

end

# Wrapper to handle several time-series of observations.
function estep(model::DynamicDiscreteModel,data::Array{Array,1},w::Array{Float64,4})
	for i=1:length(data)
		estep(model,data[i],w)
	end
end