
theta0=(.65, .5)
model=HiddenMarkovModels.toymodel()
coef!(model,theta0)
data=rand(model,100,100)


thetahat=HiddenMarkovModels.eta2theta(mle(model,data))
thetahat2=HiddenMarkovModels.eta2theta(em(model,data))
@test norm(collect(thetahat)-collect(thetahat2))<1e-3


### JACOBIANS

using Calculus

data=rand(model,10)
eta0=HiddenMarkovModels.theta2eta(theta0)
@test norm(vec(Calculus.gradient(eta -> loglikelihood(model,data,eta),eta0))-vec(loglikelihood_jac(model,data,eta0)[2])) < 1e-5

data=rand(model,10,100)
eta0=HiddenMarkovModels.theta2eta(theta0)
@test norm(vec(Calculus.gradient(eta -> loglikelihood(model,data,eta),eta0))-vec(loglikelihood_jac(model,data,eta0)[2])) < 1e-5