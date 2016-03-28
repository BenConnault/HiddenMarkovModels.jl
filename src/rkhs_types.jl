### RKHS spaces
abstract AtomicRKHS

typealias NestedTuple{T1,T2} Union{T1,Tuple{Vararg{Union{T1,T2}}}}
#allows up to three nested levels of tuples of tuples of tuples
typealias RKHS NestedTuple{AtomicRKHS,NestedTuple{AtomicRKHS,NestedTuple{AtomicRKHS,AtomicRKHS}}}



# function iselement{H <: Tuple{Vararg{RKHS}}(::Type{Tuple{H,RKHS}},x::Tuple)=
# 	iselement(H1,x[1]) && iselement(H1,x[1])	
# end


### Ground space (aka. sample space) for RKHS spaces


# Points in the ground space aka sample space
immutable Point{H <: RKHS,T}
	# If in the future spaces are more than what they say at face value, 
	# points may need to carry their space with them.
	space::H   
	# point::T
	function call{H,T}(::Type{Point}, H1::H, x::T)
        @assert iselement(H1,x)
        new{H,T}(H1,x)
    end
end



### PRODUCT KERNELS

# (1) If repeated construction of Point() is a computational bottleneck, 
# I might be able to design something with only Point{H<:AtomicRKHS}
# and Tuple{Vararg{Point}}

# (2) If points carry their space, I will have to work with:
	# x1 = Point(x.space[1],x.point[1])

function kernel{H <: Tuple{Vararg{RKHS}}}(x::Point{H},y::Point{H})
	x1 = Point(x.space[1],x.point[1])
	y1 = Point(y.space[1],y.point[1])
	if length(x.point)>1
		x2 = Point(x.space[2:end],x.point[2:end])
		y2 = Point(y.space[2:end],y.point[2:end])
		return kernel(x1,y1)*kernel(x2,y2)
	else
		return kernel(x1,y1)
	end
end

#STOPGAP
iselement(H::RKHS,x)=true



# # For kernel(Point,Point,..) calls (not usual I would think)
# function kernel{T <: Tuple{Vararg{Point}}}(xx::T,yy::T)
# 	@assert length(xx)==length(yy)
# 	n=length(xx)
# 	res=1.0
# 	for i=1:n
# 		#might be able to use a fancier one-liner but this will do:
# 		res*=kernel(xx[i],yy[i])		
# 	end
# 	res
# end


# Vectors in RKHS spaces are represented as linear combinations of basis vectors.
# As of now only k_x are allowed to be basis vectors.
# As a consequence they can be stored as `x` from the ground space.
# Note that k_x is also the embedding for delta_x, the Dirac measure at x. 
immutable RKHSVector{H <: RKHS,T}
	#Do I want abstract vectors (sparsity...) and abstract numbers (ForwardDiff...)?
	weights::Vector{Float64}    
	basis::Vector{Point{H,T}}
	function RKHSVector(weights,basis)
		@assert length(weights)==length(basis)
		new(weights,basis)
	end
end

length(v::RKHSVector)=length(v.weights)


immutable RKHSMap{H1 <: RKHS,H2 <: RKHS,T1,T2}
	leftbasis::Vector{Point{H1,T1}}
	weights::Matrix{Float64}    
	rightbasis::Vector{Point{H2,T2}}
	function RKHSVector{H1,H2,T1,T2}(leftbasis::Vector{Point{H1,T1}},weights,rightbasis::Vector{Point{H2,T2}})
		@assert size(weights)=(length(leftbasis),length(rightbasis))
		new{H1,H2,T1,T2}(leftbasis,weights,rightbasis)
	end
end



abstract Distance
immutable KernelDistance <: Distance end

#This is to compute d(delta_x,delta_yy) = d(x,y) by abuse of notation.
#Two cases: xx is a Point or xx is a Tuple{Vararg{Point}}
#In both cases xx and yy must have the same type
#I don't dispatch here on Union{Point,Tuple{Vararg{Point}}}, `kernel()` will make the proper checks downstream
evaluate(dk::KernelDistance,xx,yy)=sqrt(kernel(xx,xx)+kernel(yy,yy)-kernel(xx,yy))



# concrete RKHS


	# For now a space type carries all the relevant information in the type.
	# This way a Point{H} carries all the information in its type
	immutable GaussianRKHS{Dimension,Precision,Label} <: AtomicRKHS
		# dimension::Int
		# precision::Float64
	end

	dimension{N}(H::GaussianRKHS{N})=N
	precision{N,P}(H::GaussianRKHS{N,P})=P

	iselement(H::GaussianRKHS,x::Vector{Float64})=(length(x)==dimension(H))

	kernel{H <: GaussianRKHS}(x::Point{H},y::Point{H})=exp(-precision(x.space)*norm(x.point-y.point)/2)

	immutable DiscreteRKHS{Label} <: AtomicRKHS end

	dimension(H::GaussianRKHS)=1
	iselement(H::DiscreteRKHS,x::Int)=true

	kernel{H <: DiscreteRKHS}(x::Point{H},y::Point{H})=1.0*(x.point==y.point)

#