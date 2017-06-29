# type LinearGaussianSSM{T} <: AbstractLinearGaussian

abstract type StrictHiddenMarkovModel
end

struct LinearGaussian <: StrictHiddenMarkovModel
    sqrtv::Matrix{Float64}
    sqrtw::Matrix{Float64}
    core::LinearGaussianSSM{Float64}
end

# # from Kalman.jl
# type LinearGaussianSSM{T} <: AbstractLinearGaussian
#   F::Matrix{T}    # process matrix
#   V::Matrix{T}    # process variance
#   G::Matrix{T}    # observation matrix
#   W::Matrix{T}    # observation variance

randa(model::LinearGaussian,x)=model.core.F*x + model.sqrtv * randn(length(x))
randb(model::LinearGaussian,x)=model.core.G*x + model.sqrtw * randn(size(model.core.G,1))
    


function rand(model::LinearGaussianSSM,ini::AbstractVector,T::Int)
    dx,dy=size(model.G)
    @assert length(ini)==dx
    xx=[zeros(dx) for _=1:T]
    yy=[zeros(dx) for _=1:T]
    xx[1][:]=ini
    yy[1][:]=rand(MvNormal(model.G*xx[1],model.W))
    for t=2:T
        xx[t][:]=rand(MvNormal(model.F*xx[t-1],model.V))
        yy[t][:]=rand(MvNormal(model.G*xx[t],model.W))
    end
    xx,yy
end

rand(model::LinearGaussian,ini::AbstractVector,T::Int)=rand(model.core,ini,T)
rand(model::LinearGaussian,ini::Number,T::Int)=rand(model,[ini],T)


function rand(model::LinearGaussian,T::Int)
    dx=size(model.core.F,1)
    ini=rand(MvNormal(zeros(dx),model.core.V))
    rand(model,ini,T)
end




function lgmodel()
    pm = [1 -.2; .5 .5]
    pc = [1 0.2; 0.2 1]         # process variance
    om = eye(2)     # observation model parameter
    oc = eye(2)      # observation variance
    LinearGaussian(real(sqrtm(pc)::Matrix{Float64}),real(sqrtm(oc)),LinearGaussianSSM(pm, pc, om, oc))
end

function lgmodel(n)
    p=randn(n,n)
    pm = p\(diagm((rand(n)-.5)*1.6)*p)     # process model parameter
    pc = [0.5^abs(i-j)*sqrt(0.8^(i+j)) for i=1:n,j=1:n]         # process variance
    om = eye(n)     # observation model parameter
    oc = 0.1*eye(n)     # observation variance
    LinearGaussian(real(sqrtm(pc)),real(sqrtm(oc)),LinearGaussianSSM(pm, pc, om, oc))
end

