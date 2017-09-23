axx     = [0.9  0.1; 0.2 0.7]
sqrtvxx = [0.8  0.2; 0.2 0.7] 
axy     = [1.2 -0.2; 0.2 1.0] 
sqrtvxy = [1.3  0.2; 0.2 0.9]

T       = 5

ini_filter_mean = zeros(2)
ini_filter_vcov = eye(2)

data=HMM.unmat([
0.0   0.354819  -0.61903  -1.40201   -0.0520431;
0.0  -0.153504  -2.51111   0.421037  -0.758554
])

true_filter_mean=[
0.0   0.119875  -0.658869  -0.643061  -0.469718;
0.0  -0.112602  -1.26136   -0.226636  -0.472556
]

true_filter_vcov=reshape([
1.0  0.675745  0.610077  0.594934  0.591425;
0.0  0.180165  0.201559  0.204428  0.204696;
0.0  0.180165  0.201559  0.204428  0.204696;
1.0  0.420565  0.36862   0.362919  0.362281
],2,2,T)

linear_gaussian_model = HMM.LinearGaussian(axx,sqrtvxx,axy,sqrtvxy)
ini_filter = (ini_filter_mean, ini_filter_vcov)
filter_mean, filter_vcov = filtr(linear_gaussian_model,data,ini_filter)

ini_xy  = (zeros(2),zeros(2))
xx, yy  = rand(linear_gaussian_model,ini_xy,T)

tol=1e-4
@test norm(HMM.mat(HMM.unmat(HMM.mat(xx)))-HMM.mat(xx)) < tol
@test norm(HMM.mat(HMM.unmat(HMM.mat(yy)))-HMM.mat(yy)) < tol

@test norm(filter_mean - true_filter_mean) < tol
@test norm(vec(filter_vcov - true_filter_vcov)) < tol