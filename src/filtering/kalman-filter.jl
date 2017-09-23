

function _filtr(model::LG,data,ini,filtering_method::KalmanFilter)
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

    print("Running the Kalman filter... ")
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

