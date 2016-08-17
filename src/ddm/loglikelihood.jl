#Evaluating the likelihood requires only filtering ("forward").

# WRAPPER FUNCTIONS


#data: several iid individuals
function loglikelihood{S<:Array}(model::DynamicDiscreteModel,data::Array{S,1})
	n=length(data)
	T=0.0
	llk=0.0
	for i=1:n
		Ti=float(length(data[i]))
		llk+=loglikelihood(model,data[i])*Ti
		T+=Ti
	end
	llk/T
end



#data: several iid individuals
function loglikelihood_jac{S<:Array}(model::DynamicDiscreteModel,data::Array{S,1})
	dtheta=size(model.rhojac)
	n=length(data)
	T=0.0
	llk=0.0
	llkjac=zeros(dtheta)
	for i=1:n
		Ti=float(length(data[i]))
		temp=loglikelihood_jac(model,data[i])
		llk+=temp[1]*Ti
		llkjac+=temp[2]*Ti
		T+=Ti
	end
	llk/T,llkjac/T
end



# CORE NON-JACOBIAN METHODS

function filterstep!(model::DynamicDiscreteModel,iy,jy)
	dx=length(model.phi)

	#jx=1
	#phi is normalized from the previous step
	model.phi[1]=model.psi[1]/model.rho[1]
	model.psi[1]=model.m[1,iy,1,jy]*model.phi[1]
	for ix=2:dx
		model.phi[ix]=model.psi[ix]/model.rho[1]
		model.psi[1]+=model.m[ix,iy,1,jy]*model.phi[ix]
	end
	model.rho[1]=model.psi[1]

	for jx=2:dx
		model.psi[jx]=model.m[1,iy,jx,jy]*model.phi[1]
		for ix=2:dx
			model.psi[jx]+=model.m[ix,iy,jx,jy]*model.phi[ix]
		end
		model.rho[1]+=model.psi[jx]
	end
end

function loglikelihood(model::DynamicDiscreteModel,data::Array{Int,1})
	lambda=log(sum(model.mu[:,data[1]]))				#log-normalization factor for numerical stability

	model.psi[:]=model.mu[:,data[1]]/sum(model.mu[:,data[1]])		#actual filter
	model.rho[1]=1

	for t=2:length(data)
		filterstep!(model,data[t-1],data[t])
		lambda+=log(model.rho[1])
	end
	lambda/length(data)
end




# CORE JACOBIAN METHODS

function filterstep_jac!(model::DynamicDiscreteModel,iy,jy)
	dx,dtheta=size(model.phijac)

	#jx=1
	#phi is normalized from the previous step
	model.phi[1]=model.psi[1]/model.rho[1]
	model.psi[1]=model.m[1,iy,1,jy]*model.phi[1]
	for itheta=1:dtheta
		model.phijac[1,itheta]=(model.psijac[1,itheta]-model.phi[1]*model.rhojac[itheta])/model.rho[1]
		model.psijac[1,itheta]=model.mjac[1,iy,1,jy,itheta]*model.phi[1]+model.m[1,iy,1,jy]*model.phijac[1,itheta]
	end
	for ix=2:dx
		model.phi[ix]=model.psi[ix]/model.rho[1]
		model.psi[1]+=model.m[ix,iy,1,jy]*model.phi[ix]
		for itheta=1:dtheta
			model.phijac[ix,itheta]=(model.psijac[ix,itheta]-model.phi[ix]*model.rhojac[itheta])/model.rho[1]
			model.psijac[1,itheta]+=model.mjac[ix,iy,1,jy,itheta]*model.phi[ix]+model.m[ix,iy,1,jy]*model.phijac[ix,itheta]
		end
	end
	model.rho[1]=model.psi[1]
	for itheta=1:dtheta
		model.rhojac[itheta]=model.psijac[1,itheta]
	end


	for jx=2:dx
		model.psi[jx]=model.m[1,iy,jx,jy]*model.phi[1]
		for itheta=1:dtheta
			model.psijac[jx,itheta]=model.mjac[1,iy,jx,jy,itheta]*model.phi[1]+model.m[1,iy,jx,jy]*model.phijac[1,itheta]
		end
		for ix=2:dx
			model.psi[jx]+=model.m[ix,iy,jx,jy]*model.phi[ix]
			for itheta=1:dtheta
				model.psijac[jx,itheta]+=model.mjac[ix,iy,jx,jy,itheta]*model.phi[ix]+model.m[ix,iy,jx,jy]*model.phijac[ix,itheta]
			end
		end
		model.rho[1]+=model.psi[jx]
		for itheta=1:dtheta
			model.rhojac[itheta]+=model.psijac[jx,itheta]
		end
	end
end


function loglikelihood_jac(model::DynamicDiscreteModel,data::Array{Int,1})
	lambda=log(sum(model.mu[:,data[1]]))				#log-normalization factor for numerical stability
	dx,dtheta=size(model.phijac)
	lambdajac=zeros(dtheta)

	model.psi[:]=model.mu[:,data[1]]/sum(model.mu[:,data[1]])		#actual filter
	model.rho[1]=1
	model.psijac[:]=zeros(dx,dtheta)		#actual filter
	model.rhojac[:]=zeros(dtheta)

	for t=2:length(data)
		filterstep_jac!(model,data[t-1],data[t])
		lambda+=log(model.rho[1])
		for itheta=1:dtheta
			lambdajac[itheta]+=model.rhojac[itheta]/model.rho[1]
		end
	end
	lambda/length(data),lambdajac/length(data)
end	



#non efficient log-likelihood + jacobian implementation. keep around for test purposes
function loglikelihood_jac_nonefficient(model::DynamicDiscreteModel,data::Array{Int,1})
	dx,dy=size(model.mu)
	dtheta=size(model.mjac)[5] 					 #dimension of the parameter
	rho=log(sum(model.mu[:,data[1]]))				#log-normalization factor for numerical stability
	rhojac=zeros(1,dtheta)
	phi=model.mu[:,data[1]]/sum(model.mu[:,data[1]])		#actual filter
	phijac=zeros(dx,dtheta)
	phihatjac=zeros(dx,dtheta)

	for t=2:length(data)
		mm=reshape(model.m[:,data[t-1],:,data[t]],(dx,dx))'
		phihat=mm*phi
		for itheta=1:dtheta
			mmjac=reshape(model.mjac[:,data[t-1],:,data[t],itheta],(dx,dx))'
			phihatjac[:,itheta]=mm*phijac[:,itheta]+mmjac*phi
		end
		lambda=sum(phihat)
		lambdajac=sum(phihatjac,1)
		phi=phihat/lambda
		rho+=log(lambda)
		for itheta=1:dtheta
			phijac[:,itheta]=(phihatjac[:,itheta]*lambda - phihat*lambdajac[itheta])/lambda^2
			rhojac[itheta]+=lambdajac[itheta]/lambda
		end
	end
	rho,rhojac
end
