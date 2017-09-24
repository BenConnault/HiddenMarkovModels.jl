import ForwardDiff

type TestModel <: HMM.DynamicDiscreteModel

    #DynamicDiscreteModel fields    
    m::Array{Float64,4}             #the transition matrix given as m[x,y,x',y'] 
    mu::Array{Float64,2}            #initial distribution (dx,dy)

    #DynamicDiscreteModel's technical fields    
    mjac::Array{Float64,5}          #jacobian
    rho::Array{Float64,1}
    phi::Array{Float64,1}   
    psi::Array{Float64,1}
    rhojac::Array{Float64,1}                
    phijac::Array{Float64,2}            
    psijac::Array{Float64,2}
end

dx=2
dy=3


testmodel()=TestModel(
    Array{Float64}(dx,dy,dx,dy),             #m
    fill(1/6,2,3),                          #mu
    Array{Float64}(dx,dy,dx,dy,2),           #mjac
    Array{Float64}(1),                       #rho
    Array{Float64}(dx),                      #phi
    Array{Float64}(dx),                      #psi
    Array{Float64}(2),                       #rhojac
    Array{Float64}(dx,2),                    #phijac
    Array{Float64}(dx,2)                     #psijac
    )

function HMM.coef!(model::TestModel,theta::Tuple)
    p1,p2=theta[1],theta[2]
    a=[p1 1-p1;1-p1 p1]
    p3=(1-p2)/2
    b=[p2 p3 p3; p3 p3 p2]
    HMM.hmm2ddm!(model,a,b)
end

theta2eta(theta::Tuple)=[log(theta[1]/(1-theta[1])),log(theta[2]/(1-theta[2]))]
eta2theta(eta::Array)=(exp(eta[1])/(1+exp(eta[1])),exp(eta[2])/(1+exp(eta[2])))
HMM.coef!(model::TestModel,eta::Array)=HMM.coef!(model,eta2theta(eta))

HMM.dim(model::TestModel)=2

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
function HMM.coef_jac!(model::TestModel,eta)
    a,b=eta2ab(eta)
    fa(eta)=eta2ab(eta)[1]
    fb(eta)=eta2ab(eta)[2]
    ajac=reshape(ForwardDiff.jacobian(fa,eta),2,2,2)
    bjac=reshape(ForwardDiff.jacobian(fb,eta),2,3,2)
    HMM.hmm2ddm!(model,reshape(a,2,2),reshape(b,2,3),ajac,bjac)
end


theta0=(.65, .5)
model=testmodel()


########################################################################################################
###
########################################################################################################



HMM.coef!(model,theta0)
data=rand(model,100,100)



### JACOBIANS

using Calculus


data=rand(model,10)
eta0=HiddenMarkovModels.theta2eta(theta0)
@test norm(vec(Calculus.gradient(eta -> HMM.loglikelihood(model,data,eta),eta0))-vec(HMM.loglikelihood_jac(model,data,eta0)[2])) < 1e-5
data=rand(model,10,100)
eta0=HiddenMarkovModels.theta2eta(theta0)
@test norm(vec(Calculus.gradient(eta -> HMM.loglikelihood(model,data,eta),eta0))-vec(HMM.loglikelihood_jac(model,data,eta0)[2])) < 1e-5



########################################################################################################
###
########################################################################################################


a=[.4 .6; .3 .7]
b=[.3 .1 .6; .5 .2 .3]
data=[1,2,3]
model=hmm((a,b))

function naiveprob(data)
    fil=transpose(model.mu[:,data[1]])
    for t=2:length(data)
        fil=sum(fil*view(model.m,:,data[t-1],:,data[t]),1)
    end
    sum(fil)
end

@test vec(sum(model.m,(3,4)))≈fill(1,6) 
@test reshape(sum(view(model.m,:,1,:,:),3),2,2)≈a
@test HMM.loglikelihood(model,data)≈log(naiveprob(data))/length(data)
data=Vector{Vector{Int}}(2)
data[1]=[1,2,3]
data[2]=[3,2,1]
@test loglikelihood(model,data)≈(log(naiveprob(data[1]))/length(data[1])+log(naiveprob(data[2]))/length(data[2]))/2

viterbi(model,data)

data=rand(model,10000)
abhat=em(model,data)

@test vecnorm(abhat[1]-a)<5e-2
@test vecnorm(abhat[2]-b)<5e-2