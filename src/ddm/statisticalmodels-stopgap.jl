import StatsBase.StatisticalModel
using Optim

import Distributions: loglikelihood, dim

export coef!, coef_jac!, rand, loglikelihood, loglikelihood_jac, dim, mle

#fall-back error messages. This is a way of enforcing the interface.
coef!(model::StatisticalModel,parameter)=error("Method coef!(model::$(typeof(model)),parameter) must be implemented for $(typeof(model)).")
coef_jac!(model::StatisticalModel,parameter)=error("Method coef_jac!(model::$(typeof(model)),parameter) must be implemented for $(typeof(model)).")
rand(model::StatisticalModel,T::Int)=error("Method rand(model::$(typeof(model)),T::Int) must be implemented for $(typeof(model)).")
loglikelihood(model::StatisticalModel,data)=error("Method loglikelihood(model::$(typeof(model)),data) must be implemented for $(typeof(model)).")
loglikelihood_jac(model::StatisticalModel,data)=error("Method loglikelihood_jac(model::$(typeof(model)),data) must be implemented for $(typeof(model)).")


#return the dimension of the parameter: used to provide a random starting value when calling mle()
dim(model::StatisticalModel)=error("If you want to use the default MLE method on a $(typeof(model)), you must define dim(model::$(typeof(model))).")

function loglikelihood(model::StatisticalModel,data,parameter)
	coef!(model,parameter)
	loglikelihood(model,data)
end

function loglikelihood_jac(model::StatisticalModel,data,parameter)
	coef_jac!(model,parameter)
	loglikelihood_jac(model,data)
end

function mle_nojac(model::StatisticalModel,data,thetai::Array{Float64,1}=rand(dim(model)),L=15000)
	fcalls=0
	function ff(theta)
		fcalls+=1
		llk=-loglikelihood(model,data,theta)
		# println(fcalls,": ",round(theta,3),", ",round(llk,3))
		llk
	end
	df=Optim.DifferentiableFunction(ff)	
	ret = Optim.optimize(df, thetai,method=ConjugateGradient(),iterations=L)
	println("  no jacobian, $(ret.iterations) iterations, $fcalls ff evaluations, final log-likelihood: $(round(-ret.f_minimum,4))")
	ret.minimum
end

function mle_jac(model::StatisticalModel,data,thetai::Array{Float64,1}=rand(dim(model)),L=15000)
	fcalls=0
	fjcalls=0
	ffjcalls=0
	function ff(theta)
		fcalls+=1
		-loglikelihood(model,data,theta)
	end
	function fj!(theta,jac)
		fjcalls+=1
		jac[:]=-loglikelihood_jac(model,data,theta)[2]
	end
	function ffj!(theta,jac)
		ffjcalls+=1
		res=loglikelihood_jac(model,data,theta)
		jac[:]=-res[2]
		-res[1]
	end
	df=Optim.DifferentiableFunction(ff,fj!,ffj!)
	ret = Optim.optimize(df, thetai,method=ConjugateGradient(),iterations=L)
	println("  with jacobian, $(ret.iterations) iterations, ($fcalls,$fjcalls,$ffjcalls) (ff,fj,ffj) evaluations, final log-likelihood: $(round(-ret.f_minimum,4))")
	ret.minimum
end

function mle(model::StatisticalModel,data,thetai::Array{Float64,1}=rand(dim(model)),L=15000)
	println()
	println(" generic optimization of the likelihood...")
	try
		mle_jac(model,data,thetai,L)
	catch
		mle_nojac(model,data,thetai,L)
	end
end

function fit!(model::StatisticalModel,data)
	thetahat=mle(model,data)
	coef!(model,thetahat)
end
