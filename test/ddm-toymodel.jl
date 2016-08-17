#need to `importall` rather than `using` because we will extend `coef!()` and `dim()`
importall HiddenMarkovModels 

import ForwardDiff

type ToyModel <: DynamicDiscreteModel

	#DynamicDiscreteModel fields	
	m::Array{Float64,4}			  	#the transition matrix given as m[x,y,x',y'] 
	mu::Array{Float64,2}  			#initial distribution (dx,dy)

	#DynamicDiscreteModel's technical fields	
	mjac::Array{Float64,5}			#jacobian
	rho::Array{Float64,1}
	phi::Array{Float64,1}	
	psi::Array{Float64,1}
	rhojac::Array{Float64,1}				
	phijac::Array{Float64,2}			
	psijac::Array{Float64,2}
end

dx=2
dy=3


toymodel()=ToyModel(
	Array(Float64,dx,dy,dx,dy), 			#m
	fill(1/6,2,3), 							#mu
	Array(Float64,dx,dy,dx,dy,2),			#mjac
	Array(Float64,1), 						#rho
	Array(Float64,dx),						#phi
	Array(Float64,dx),						#psi
	Array(Float64,2), 						#rhojac
	Array(Float64,dx,2),					#phijac
	Array(Float64,dx,2)						#psijac
	)

function coef!(model::ToyModel,theta::Tuple)
	p1,p2=theta[1],theta[2]
	a=[p1 1-p1;1-p1 p1]
	p3=(1-p2)/2
	b=[p2 p3 p3; p3 p3 p2]
	hmm2ddm!(model,a,b)
end

theta2eta(theta::Tuple)=[log(theta[1]/(1-theta[1])),log(theta[2]/(1-theta[2]))]
eta2theta(eta::Array)=(exp(eta[1])/(1+exp(eta[1])),exp(eta[2])/(1+exp(eta[2])))
coef!(model::ToyModel,eta::Array)=coef!(model,eta2theta(eta))

dim(model::ToyModel)=2

function eta2ab(eta)
	theta=eta2theta(eta)
	a=[theta[1],1-theta[1],1-theta[1],theta[1]]
	p2=theta[2]
	p3=(1-p2)/2
	b=[p2, p3, p3, p3, p3, p2]
	(a,b)
end

#quick implementation of derivatives with numerical gradient
#explicit expressions should be easy to implement
function coef_jac!(model::ToyModel,eta)
	a,b=eta2ab(eta)
	fa(eta)=eta2ab(eta)[1]
	fb(eta)=eta2ab(eta)[2]
	ajac=reshape(ForwardDiff.jacobian(fa)(eta),2,2,2)
	bjac=reshape(ForwardDiff.jacobian(fb)(eta),2,3,2)
	hmm2ddm!(model::DynamicDiscreteModel,reshape(a,2,2),reshape(b,2,3),ajac,bjac)
end


