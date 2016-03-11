
######################################
### CANONICAL DISCRETE RKHS
######################################

immutable DiscreteRKHS{T} <: RKHS end

typealias HD DiscreteRKHS

kernel{T<:DiscreteRKHS}(::Type{T},x1::Int,x2::Int)=1.0*(x1==x2)



immutable GaussianRKHS{N} <: RKHS 
	precision::Float64    #inverse variance
end

typealias HG GaussianRKHS

# kernel(H::GaussianRKHS,x1::AbstractVector,x2::AbstractVector)=exp(-H.precision*norm(x1-x2)/2)

kernel{H<:GaussianRKHS}(::Type{H},x1::AbstractVector,x2::AbstractVector)=exp(-norm(x1-x2)/2)
kernel{H<:GaussianRKHS}(::Type{H},x1::Number,x2::Number,precision=1)=exp(-precision*norm(x1-x2)/2)
