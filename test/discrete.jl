######################################################################################
### Corner case tests
######################################################################################

struct SimpleModel{F} <: HMM.DiscreteHMM
    nx::Int
    ny::Int
    qxx::Matrix{F}
    qxy::Matrix{F}
end

function simple_model(qxx::Matrix,qxy::Matrix)
    nx,ny = size(qxy)
    SimpleModel(nx,ny,qxx,qxy)
end

HMM.qxyxy(model::SimpleModel,x,y,x2,y2) = model.qxx[x,x2]*model.qxy[x2,y2]


corner_qxx = ones(1,1)
corner_qxy = [1. 0]

corner_model = simple_model(corner_qxx,corner_qxy)

@test rand(corner_model,(1,1),5) == (ones(Int,5), ones(Int,5))








######################################################################################
### Test data
# example based on code from github/ahwillia
# > based on Michael Hamilton's "dishonest casino" example:
# > http://www.cs.colostate.edu/~hamiltom/code.html#python-hidden-markov-model
######################################################################################


A = [ 0.99 0.01 ;
      0.01 0.99 ]
B = [ 1/6   1/6   1/6   1/6   1/6   1/6 ;
      1/10  1/10  1/10  1/10  1/10  1/2 ]
ini = [0.625, 0.375]

data = [1, 2, 1, 6, 6] # observation sequence



true_filter = [ 
0.625   0.7332  0.8173  0.5884  0.3212 ;
0.375   0.2668  0.1827  0.4116  0.6788 
]

true_llk = -1.3044969853

true_smoother = [
0.381077  0.372942  0.356877  0.328592  0.321181 ;
0.618923  0.627058  0.643123  0.671408  0.678819
]

# joint_smoother[:,:,t] contains p(x_t,x_{t+1}|y_{1:T})
true_joint_smoother = reshape([
0.370695    0.35557     0.327852     0.318927;
0.00224664  0.00130683  0.000740323  0.00225305;
0.0103818   0.0173715   0.0290255    0.00966447;
0.616676    0.625751    0.642383     0.669155
],2,2,4)

######################################################################################
### Native discrete filtering: what a user should use
######################################################################################




model_2 = simple_model(A,B)

llk_from_native = loglikelihood(model_2,ini,data)
filter_from_native,smoother_from_native = filter_smoother(model_2,ini,data)
joint_smoother_from_native = HMM._smoother(model_2,filter_from_native,data)[2]

######################################################################################
### Discrete filtering from legacy code at src/ddm/
######################################################################################

model_1 = HMM.dhmm((A,B),ini*fill(1/6,1,6))
fil,smo,cond=filtr(model_1,data)  # from DynamicDiscreteModels.jl
filter_from_legacy   = fil


llk_from_legacy = loglikelihood(model_1,data)
joint_smoother_from_legacy = cond[:,:,1:4]


######################################################################################
### Kernel filtering with regularization parameter = 0
###    - continuous approximate nonlinear filtering should recover exact discrete filtering  
###    - tests both discrete and nonlinear filtering
######################################################################################

struct FakeDSHM <: HMM.StrictHMM end
model_3=FakeDSHM()

bxx=[1,2]
byy=[1,2,3,4,5,6]

kf=HMM.KKF(model_3,bxx,byy,A,B,0.0)

filter_from_fake=HMM._filtr(model_3,ini,data,kf)

######################################################################################
### Tests
######################################################################################


tol=1e-3

@test norm(vec(filter_from_legacy   - true_filter  ))   < tol 
@test norm(vec(filter_from_native   - true_filter  ))   < tol 
@test norm(vec(filter_from_fake   - true_filter    ))   < tol 

@test norm(llk_from_native - true_llk)   < tol 
@test norm(llk_from_legacy - true_llk)   < tol 

@test norm(vec(smoother_from_native   - true_smoother  ))   < tol 

@test norm(vec(joint_smoother_from_legacy   - true_joint_smoother  ))   < tol 
@test norm(vec(joint_smoother_from_native   - true_joint_smoother  ))   < tol 



