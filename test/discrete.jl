A = [ 0.99 0.01 ;
      0.01 0.99 ]
B = [ 1/6   1/6   1/6   1/6   1/6   1/6 ;
      1/10  1/10  1/10  1/10  1/10  1/2 ]
ini = [0.625, 0.375]

data = [1, 2, 1, 6, 6] # observation sequence


true_filter= [ 0.625   0.7332  0.8173  0.5884  0.3212 ;
                 0.375   0.2668  0.1827  0.4116  0.6788 ]


### Good old code from 2014....
model_1 = hmm((A,B),ini*fill(1/6,1,6))
fil,smo,cond=filtr(model_1,data)  # from DynamicDiscreteModels.jl
filter_from_old=fil

### Native discrete filter
model_2 = HMM.DiscreteStrictHiddenMarkov(A,B)
filter_from_native = filtr(model_2,data,ini)

### Pretend that the discrete model is a continuous one and apply a nonlinear filtering technique
#   with regularization parameter = 0. The results should be the same.
#   This is not what a user should do so we have to make a couple of awkward calls prefixed with `HMM.`,
#   but it is a good test to have. If the results are indeed the same, it increases the chances that both 
#   the discrete and the continuous algorithms are correctly implemented!

struct FakeDSHM <: HMM.StrictHiddenMarkov end
model_3=FakeDSHM()

bxx=[1,2]
byy=[1,2,3,4,5,6]


kf=HMM.KKF(model_3,bxx,byy,A,B,0.0)

filter_from_fake=HMM._filtr(model_3,data,ini,kf)



tol=1e-3

@test norm(vec(filter_from_old   - true_filter     ))   < tol 
@test norm(vec(filter_from_native   - true_filter  ))   < tol 
@test norm(vec(filter_from_fake   - true_filter    ))   < tol 




