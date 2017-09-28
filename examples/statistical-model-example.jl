# BC 09/2017
# Simple fully-discrete hidden Markov model parametrized with a two-dimensional statistical parameter.
# Illustrates:
# - how to extend the package's abstract type `DiscreteHMM` in order to implement a concrete statistical model.
# - the package's native methods: `filtr()`, `filter_smoother()`, `loglikelihood()`.
# - how to compute the MLE using an external optimization package. You can either do plain maximization over `loglikelihood()`, 
#   or use the EM-algorithm with E-step via `eweights!()` and M-step via numerical optimization. 
# - `likelihood()` is compatible with automatic differentiation   


module dev

using HiddenMarkovModels
HMM=HiddenMarkovModels

using Distributions.wsample


######################################################################################
### Model definition
###   
### - the model has "strict hidden Markov dynamics": Q(x',y'|x,y) = Q_{xx}(x'|x)Q_{xy}(y'|x')
### - the model is parametrized by a 2-dimensional statistical parameter, θ ∈ [0,1]^2.
###   Specifically:
###    (i) Q_{xx} = Q_{xx}(θ_1) = θ_1 Q_{xx}^1 + (1-θ_1) Q_{xx}^2  
###   (ii) Q_{xy} = Q_{xy}(θ_2) = θ_2 Q_{xy}^1 + (1-θ_2) Q_{xy}^2  
### 
### - we simulate data from the model with true parameter value θ* = (0.2,0.7).
### - based on the data, we try and estimate θ
### 
######################################################################################


struct SimpleModel{F} <: HMM.DiscreteHMM
    theta::Vector{F}
    qxx::Matrix{F}
    qxy::Matrix{F}
end



dx,dy = 2,2

qxx1 = [0.9 0.1; 0.1 0.9]
qxx2 = [0.1 0.9; 0.9 0.1]
qxy1 = [0.95 0.05; 0.5 0.5]
qxy2 = [0.5 0.5; 0.05 0.95] 

function coef!(model::SimpleModel,theta)
    model.theta[:] = theta
    model.qxx[:] = theta[1]*qxx1 + (1-theta[1])*qxx2
    model.qxy[:] = theta[2]*qxy1 + (1-theta[2])*qxy2
end

function simple_model(theta)
    model = SimpleModel(theta,zeros(eltype(theta),dx,dx),zeros(eltype(theta),dx,dy))
    coef!(model,theta)
    model
end

### Implementing the HMM.DiscreteHMM interface:

function HMM.rand(model,xy)
    x,y = xy
    x2  = wsample(view(model.qxx,x,:))
    y2  = wsample(view(model.qxy,x2,:))
    x2,y2
end

HMM.qxyxy(model::SimpleModel,x,y,x2,y2) = model.qxx[x,x2]*model.qxy[x2,y2]




######################################################################################
### Picking a true parameter value and simulating some data
######################################################################################

true_theta = [0.2, 0.7]  
my_model   = simple_model(copy(true_theta))
true_qxx   = my_model.qxx
true_qxy   = my_model.qxy

initial = (1,1)
T = 100_000
xx,yy   = rand(my_model,initial,T)

######################################################################################
### Parameter estimation by MLE
### We use Optim.jl's BFGS solver together with ForwardDiff's automatic differentiation.
### (`HiddenMarkovModels.jl` is compatible with automatic differentiation.)
######################################################################################

initial_filter = [1.,0]


using Optim, ForwardDiff

sigmo(x) = 1/(1+exp(-x))
sigmi(q) = log(q/(1-q))


function mle(model::SimpleModel, initial_filter, data)
    

    function ff(x)
        theta=sigmo.(x)
        coef!(model,theta)
        -loglikelihood(model,initial_filter,data)
    end

    function llk(x)
        theta = sigmo.(x)
        ad_model = simple_model(theta)
        -loglikelihood(ad_model,initial_filter,data)
    end

    fg!(grad, x) = ForwardDiff.gradient!(grad,llk,x)

    initial_x = sigmi.(true_theta)

    prog = optimize(ff,fg!,initial_x,BFGS())

    sigmo.(prog.minimizer)
end

theta_mle = mle(my_model,initial_filter,yy) 

coef!(my_model,true_theta)
true_llk  = loglikelihood(my_model,initial_filter,yy)

coef!(my_model,theta_mle)
llk_mle  = loglikelihood(my_model,initial_filter,yy)

println()
println("true θ:")
println(round.(true_theta,3))

println()
println("estimated θ:")
println(round.(theta_mle,3))





end #module


