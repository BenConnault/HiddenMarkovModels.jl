type HiddenMarkovModel <: DynamicDiscreteModel
	#HMM specific field

	#DynamicDiscreteModel fields
	m::Array{Float64,4}			  	#the transition matrix given as m[x,y,x',y'] 
	mu::Array{Float64,2}  			#initial distribution (dx,dy)
	rho::Array{Float64,1}
	phi::Array{Float64,1}	
	psi::Array{Float64,1}
end

function hmm(ab::Tuple,mu)
	a,b=ab
	dx,dy=size(mu)
	model=HiddenMarkovModel(Array(Float64,dx,dy,dx,dy),mu,Array(Float64,1),Array(Float64,dx),Array(Float64,dx))
	hmm2ddm!(model,a,b)
	model
end

#default initial distribution uniform
function hmm(ab::Tuple)
	dx,dy=size(ab[2])
	mu=fill(1/(dx*dy),dx,dy)
	hmm(ab,mu)
end

#ab=(a,b) parametrization
calibrate!(model::HiddenMarkovModel,ab::Tuple)=hmm2ddm!(model,ab[1],ab[2])

#canonical parametrization in case someone wants to apply mle() to it
calibrate!(model::HiddenMarkovModel,theta::Array{Float64,1})=calibrate(model,theta2ab(theta,size(model.mu,1)))

theta2ab(theta::Array{Float64,1},dx::Int)=(z2q(theta[1:dx*(dx-1)]),z2q(theta[dx*(dx-1)+1:end],dx))

function dim(model::HiddenMarkovModel)
	dx,dy=size(model.mu)
	dx*(dx-1)+dx*(dy-1)
end


function em(model::HiddenMarkovModel,data)
	dx,dy=size(model.mu)
	w=zeros(dx,dy,dx,dy)
	estep(model,data,w)	
	a=nsm(reshape(sum(w,(2,4)),dx,dx))
	b=nsm(reshape(sum(w,(1,2)),dx,dy))
	llk=loglikelihood(model,data,(a,b))
	println()
	println(" estimating hidden Markov model via Baum-Welch algorithm...")
	println("  log-likelihood: $(round(llk,4))")
	(a,b)
end	

baumwelch=em
