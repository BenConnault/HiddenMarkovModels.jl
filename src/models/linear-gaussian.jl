struct LinearGaussian <: StrictHiddenMarkov
    axx::Matrix{Float64}
    sqrtvxx::Matrix{Float64}
    axy::Matrix{Float64}
    sqrtvxy::Matrix{Float64}
end


struct KalmanFilter <: FilteringTechnique
end



cgaussian(a,sqrtv,x) = a*x+sqrtv*randn(length(x))

draw_x(model::LinearGaussian,x) = cgaussian(model.axx,model.sqrtvxx,x)
draw_y(model::LinearGaussian,x) = cgaussian(model.axy,model.sqrtvxy,x)

initial_filter(model::LinearGaussian,ini,filtering_method) = ini
filtr(model::LinearGaussian,data,ini) = filtr(model,data,ini,KalmanFilter())


# cpdf(model::ConditionalGaussian,x,y)=exp(-sum((model.sqrtv\(y-model.A*x)).^2)/2)/((2*pi)^(length(y)/2)*det(model.sqrtv))



# function lgmodel()
#     pm = [1 -.2; .5 .5]
#     pc = [1 0.2; 0.2 1]         # process variance
#     om = eye(2)     # observation model parameter
#     oc = eye(2)      # observation variance
#     LinearGaussian(pm, pc, om, oc)
# end

# function lgmodel(n)
#     p=randn(n,n)
#     pm = p\(diagm((rand(n)-.5)*1.6)*p)     # process model parameter
#     pc = [0.5^abs(i-j)*sqrt(0.8^(i+j)) for i=1:n,j=1:n]         # process variance
#     om = eye(n)     # observation model parameter
#     # oc = 0.1*eye(n)     # observation variance
#     oc = eye(n)     # observation variance
#     LinearGaussian(pm, pc, om, oc)
# end

