abstract type DynamicDiscreteModel
end
	# A concrete implementation of a DynamicDiscreteModel promises to implement the following fields:

	# m::Array{Float64,4}			  	#the transition matrix given as m[x,y,x',y'] 
	# mu::Array{Float64,2}  			#initial distribution (dx,dy)
	# mjac::Array{Float64,5}			#jacobian
	
	# #discrete filter variables
	# rho::Array{Float64,1}				#container for the constant rho used in the discrete filter
	# phi::Array{Float64,1}			#filter value today
	# psi::Array{Float64,1}			#filter value tomorrow


	#if you want to implement jacobian too
	# rhojac::Array{Float64,1}				
	# phijac::Array{Float64,2}			
	# psijac::Array{Float64,2}




# import StatsBase.StatisticalModel
# using Optim

# import Distributions: loglikelihood, dim

# export coef!, coef_jac!, rand, loglikelihood, loglikelihood_jac, dim, mle

### INTERFACE
# # fall-back error messages. This is a way of enforcing the interface.
coef!(model::DynamicDiscreteModel,parameter)=error("Method coef!(model::$(typeof(model)),parameter) must be implemented for $(typeof(model)).")
coef_jac!(model::DynamicDiscreteModel,parameter)=error("Method coef_jac!(model::$(typeof(model)),parameter) must be implemented for $(typeof(model)).")
# rand(model::StatisticalModel,T::Int)=error("Method rand(model::$(typeof(model)),T::Int) must be implemented for $(typeof(model)).")
# loglikelihood(model::StatisticalModel,data)=error("Method loglikelihood(model::$(typeof(model)),data) must be implemented for $(typeof(model)).")
# loglikelihood_jac(model::StatisticalModel,data)=error("Method loglikelihood_jac(model::$(typeof(model)),data) must be implemented for $(typeof(model)).")
# # return the dimension of the parameter: used to provide a random starting value when calling mle()
# dim(model::StatisticalModel)=error("If you want to use the default MLE method on a $(typeof(model)), you must define dim(model::$(typeof(model))).")


function loglikelihood(model,data,parameter)
	coef!(model,parameter)
	loglikelihood(model,data)
end

function loglikelihood_jac(model,data,parameter)
	coef_jac!(model,parameter)
	loglikelihood_jac(model,data)
end



