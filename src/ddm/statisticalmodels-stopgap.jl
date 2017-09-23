# import StatsBase.StatisticalModel
# using Optim

# import Distributions: loglikelihood, dim

# export coef!, coef_jac!, rand, loglikelihood, loglikelihood_jac, dim, mle

### INTERFACE
# # fall-back error messages. This is a way of enforcing the interface.
# # fall-back error messages. This is a way of enforcing the interface.
coef!(model::StatisticalModel,parameter)=error("Method coef!(model::$(typeof(model)),parameter) must be implemented for $(typeof(model)).")
coef_jac!(model::StatisticalModel,parameter)=error("Method coef_jac!(model::$(typeof(model)),parameter) must be implemented for $(typeof(model)).")
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

