# Notes:
# - Kalman filter/smoother works with determinstic observations (noise vcov =0)



# inheriting gives access to simulation method `rand()`
struct LinearGaussianHMM <: StrictHMM
    axx::Matrix{Float64}
    sqrtvxx::Matrix{Float64}
    axy::Matrix{Float64}
    sqrtvxy::Matrix{Float64}
end


cgaussian(a,sqrtv,x) = a*x+sqrtv*randn(size(a,1))

draw_x(model::LinearGaussianHMM,x) = cgaussian(model.axx,model.sqrtvxx,x)
draw_y(model::LinearGaussianHMM,x) = cgaussian(model.axy,model.sqrtvxy,x)

cpdf(model::LinearGaussianHMM,flag::Val{:x},x,x2) = cpdf_gaussian(x,x2,model.axx,model.sqrtvxx) 
cpdf(model::LinearGaussianHMM,flag::Val{:y},x,y)  = cpdf_gaussian(x,y,model.axy,model.sqrtvxy)

cpdf_gaussian(x::Vector{Float64},y::Vector{Float64},axy::Matrix{Float64},sqrtvxy::Matrix{Float64})::Float64 = 
        exp(-sum((sqrtvxy\(y-axy*x)).^2)/2)/((2*pi)^(length(y)/2)*det(sqrtvxy))



# ini:  (ini_mean,ini_vcov) where mu(x_1|y_1) ~ N(ini_mean,ini_vcov)
function filtr(model::LinearGaussianHMM,ini,data)
    T=length(data)

    ini_mean,ini_vcov = ini

    ny,nx = size(model.axy)
    filter_mean = Array{Float64}(nx,T)
    filter_vcov = Array{Float64}(nx,nx,T)
    
    filter_mean[:,1] = ini_mean
    filter_vcov[:,:,1] = ini_vcov

    predic_mean = zeros(nx) 
    predic_vcov = zeros(nx,nx) 
    
    tempx = zeros(nx,nx) 
    tempy = zeros(ny,nx) 
    S    = zeros(ny,ny)

    vxx = model.sqrtvxx^2
    vxy = model.sqrtvxy^2

    print("Running Kalman filter... ")
    for t=1:T-1

        # m = a mean_t
        # v = a vcvo_t a' + vxx
        A_mul_B!(predic_mean, model.axx, view(filter_mean,:,t))
        A_mul_B!(tempx, model.axx, view(filter_vcov,:,:,t))
        A_mul_Bt!(predic_vcov, tempx, model.axx)
        broadcast!(+,predic_vcov,predic_vcov,vxx)

        # s = b v b' + vxy
        # k = v b' s^-1
        A_mul_B!(tempy, model.axy, predic_vcov)
        A_mul_Bt!(S, tempy, model.axy)
        broadcast!(+,S,S,vxy)
        K = predic_vcov*At_rdiv_B(model.axy,S)
        
        filter_mean[:,t+1] = predic_mean + K*(data[t+1]-model.axy*predic_mean)
        filter_vcov[:,:,t+1] = (I-K*model.axy)*predic_vcov

    end
    println()

    filter_mean, filter_vcov
end


function _smoother(model::LinearGaussianHMM,fil,data)
    fil_mean, fil_vcov = fil
    nx, T = size(fil_mean)

    smo_mean = Array{Float64}(nx,T)
    smo_vcov = Array{Float64}(nx,nx,T)

    smo_mean[:,T]   = fil_mean[:,T]
    smo_vcov[:,:,T] = fil_vcov[:,:,T]

    vxx = model.sqrtvxx^2
    vxy = model.sqrtvxy^2

    g = zeros(nx,nx)
    temp = zeros(nx,nx)
    pred_mean = zeros(nx)
    pred_vcov = zeros(nx,nx)

    print("Running Kalman smoother... ")
    for t=T-1:-1:1
        # m = a mean_t
        # v = a vcvo_t a' + vxx
        A_mul_B!(pred_mean, model.axx, view(fil_mean,:,t))
        A_mul_B!(temp, model.axx, view(fil_vcov,:,:,t))
        A_mul_Bt!(pred_vcov, temp, model.axx)
        broadcast!(+,pred_vcov,pred_vcov,vxx)

        A_mul_B!(g,view(fil_vcov,:,:,t),At_rdiv_B(model.axx,pred_vcov))
        smo_mean[:,t]   = fil_mean[:,t]   + g*(smo_mean[:,t+1]   - pred_mean) 
        smo_vcov[:,:,t] = fil_vcov[:,:,t] + g*(smo_vcov[:,:,t+1] - pred_vcov)*g'
    end
    println()
    smo_mean, smo_vcov
end


function filter_smoother(model::LinearGaussianHMM,ini_filter,data)
    fil = filtr(model,ini_filter,data)
    smo = _smoother(model,fil,data)
    fil[1],fil[2],smo[1],smo[2]
end




