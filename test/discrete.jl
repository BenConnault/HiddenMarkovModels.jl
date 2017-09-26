######################################################################################
### Test data
######################################################################################


A = [ 0.99 0.01 ;
      0.01 0.99 ]
B = [ 1/6   1/6   1/6   1/6   1/6   1/6 ;
      1/10  1/10  1/10  1/10  1/10  1/2 ]
ini = [0.625, 0.375]

data = [1, 2, 1, 6, 6] # observation sequence


true_filter= [ 0.625   0.7332  0.8173  0.5884  0.3212 ;
                 0.375   0.2668  0.1827  0.4116  0.6788 ]



######################################################################################
### Native discrete filtering: what a user should use
######################################################################################


struct SimpleModel{F} <: HMM.DiscreteHMM
    qxx::Matrix{F}
    qxy::Matrix{F}
end

model_2 = SimpleModel(A,B)

HMM.qxyxy(model::SimpleModel,x,y,x2,y2) = model.qxx[x,x2]*model.qxy[x2,y2]

llk_from_native = loglikelihood(model_2,ini,data)
filter_from_native,smoother_from_native = filter_smoother(model_2,ini,data)


######################################################################################
### Discrete filtering from legacy code at src/ddm/
######################################################################################

model_1 = HMM.dhmm((A,B),ini*fill(1/6,1,6))
fil,smo,cond=filtr(model_1,data)  # from DynamicDiscreteModels.jl
filter_from_old   = fil
smoother_from_old = smo
llk_from_old = loglikelihood(model_1,data)

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

@test norm(vec(filter_from_old   - true_filter     ))   < tol 
@test norm(vec(filter_from_native   - true_filter  ))   < tol 
@test norm(vec(filter_from_fake   - true_filter    ))   < tol 




