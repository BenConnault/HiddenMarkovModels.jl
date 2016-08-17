# EMalgorithm() currently does not use Jacobians. If profiling shows that most of the time is spent in the M-step rather than teh E-step, 
# this might something useful to do. 

"""
Generic EM algorithm.

	em(model,data,thetai,L=1000)

Maximization of the likelihood by EM algorithm. The E-step is handled by `estep()`'s 
forward-backward algorithm and the M-step is a generic numerical optimization step. 
`thetai` is a starting value and `L` a maximum numer of iterations. 

"""
function em(model::DynamicDiscreteModel,data,thetai=rand(dim(model)),L=1000,tol=1e-10)
	dx,dy=size(model.mu)
	w=zeros(dx,dy,dx,dy)
	llks=zeros(L)
	theta=copy(thetai)
	coef!(model,theta)
	
	nonzeroindicies=find(model.m.!=0.0)

	llks[1]=loglikelihood(model,data)
	l=1
	go=true
	while go
		l+=1
		w[:]=0
		estep(model,data,w)
		function ff(xtheta)
			coef!(model,xtheta)
			#w[] has zero at lest wherever model.m has zeros, and maybe more
			#we still must be careful of not calling log(0)
			#this is what nonzeroindicies is for
			res=0.0
			for i in nonzeroindicies
				res+=w[i]*log(model.m[i])
			end
			-res
		end
		ret = Optim.optimize(ff, theta,method=:cg,iterations=L)
		theta[:]=ret.minimum
		coef!(model,theta)
		llks[l]=loglikelihood(model,data)
		go=(abs(llks[l]-llks[l-1])>tol && l<L)
	end
	println()
	println(" estimating dynamic discrete model via EM algorithm...")
	println("  $l iterations, final log-likelihood: $(round(llks[l],4))")
	# plot(llks[1:l])
	theta
end	