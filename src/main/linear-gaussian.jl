# inheriting gives access to simulation method `rand()`
struct LinearGaussianHMM <: StrictHMM
    axx::Matrix{Float64}
    sqrtvxx::Matrix{Float64}
    axy::Matrix{Float64}
    sqrtvxy::Matrix{Float64}
end


cgaussian(a,sqrtv,x) = a*x+sqrtv*randn(length(x))

draw_x(model::LinearGaussianHMM,x) = cgaussian(model.axx,model.sqrtvxx,x)
draw_y(model::LinearGaussianHMM,x) = cgaussian(model.axy,model.sqrtvxy,x)

cpdf(model::LinearGaussianHMM,flag::Val{:x},x,x2) = cpdf_gaussian(x,x2,model.axx,model.sqrtvxx) 
cpdf(model::LinearGaussianHMM,flag::Val{:y},x,y)  = cpdf_gaussian(x,y,model.axy,model.sqrtvxy)

cpdf_gaussian(x,y,axy,sqrtvxy)=exp(-sum((sqrtvxy\(y-axy*x)).^2)/2)/((2*pi)^(length(y)/2)*det(sqrtvxy))



# ini:  (ini_mean,ini_vcov) where mu(x_1|y_1) ~ N(ini_mean,ini_vcov)
function filtr(model::LinearGaussianHMM,ini,data)
    T=length(data)

    ini_mean,ini_vcov = ini

    nx = length(ini_mean)
    filter_mean = Array{Float64}(nx,T)
    filter_vcov = Array{Float64}(nx,nx,T)
    
    filter_mean[:,1] = ini_mean
    filter_vcov[:,:,1] = ini_vcov

    predic_mean = zeros(nx) 
    predic_vcov = zeros(nx,nx) 
    
    temp = zeros(nx,nx) 
    S    = zeros(nx,nx)

    vxx = model.sqrtvxx^2
    vxy = model.sqrtvxy^2

    print("Running Kalman filter... ")
    for t=1:T-1

        A_mul_B!(predic_mean, model.axx, view(filter_mean,:,t))
        A_mul_B!(temp, model.axx, view(filter_vcov,:,:,t))
        A_mul_Bt!(predic_vcov, temp, model.axx)
        broadcast!(+,predic_vcov,predic_vcov,vxx)

        A_mul_B!(temp, model.axy, predic_vcov)
        A_mul_Bt!(S, temp, model.axy)
        broadcast!(+,S,S,vxy)
        K = predic_vcov*At_rdiv_B(model.axy,S)
        
        filter_mean[:,t+1] = (I-K*model.axy)*predic_mean + K*data[t+1]
        filter_vcov[:,:,t+1] = (I-K*model.axy)*predic_vcov

    end
    println()

    filter_mean, filter_vcov
end






