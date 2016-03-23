### RKHS spaces

typealias NestedTuple{T1,T2} Union{T1,Tuple{Vargarg{Union{T1,T2}}}}
#allows up to three nested levels of tuples of tuples of tuples
typealias RKHS NestedTuple{AtomicRKHS,NestedTuple{AtomicRKHS,NestedTuple{AtomicRKHS,AtomicRKHS}}}

# concrete RKHS

	abstract AtomicRKHS

	immutable GaussianRKHS{Dimension,Precision,Label} <: AtomicRKHS
		# dimension::Int
		# precision::Float64
	end

	dimension(H::GaussianRKHS)=H.dimension
	precision(H::GaussianRKHS)=H.precision

	iselement(H::GaussianRKHS,x::Vector{Float64})=(length(x)==H.dimension)

	kernel{H <: GaussianRKHS}(x::Point{H},y::Point{H})=exp(-precision(H)*norm(x.point-y.point)/2)

	immutable DiscreteRKHS{Label} <: AtomicRKHS end

	dimension(H::GaussianRKHS)=1
	iselement(H::DiscreteRKHS,x::Int)=true

	kernel{H <: DiscreteRKHS}(x::Point{H},y::Point{H})=1.0*(x.point==y.point)

	iselement(H::AtomicRKHS,x)=false

#

function iselement{H <: Tuple{Vararg{RKHS}}(::Type{Tuple{H,RKHS}},x::Tuple)=
	iselement(H1,x[1]) && iselement(H1,x[1])	
end

### PRODUCT KERNELS

function kernel{H1 <: RKHS,H2 <: RKHS}(xx::Point{Tuple{H1,H2}},yy::Point{Tuple{H1,H2}})=
	xx1 = Point(H1,xx.point[1])
	yy1 = Point(H1,yy.point[1])
	if length(xx1)>2
		xx2 = Point(H1,xx.point[2:end])
		yy2 = Point(H1,yy.point[2:end])
	else
		xx2 = Point(H1,xx.point[2])
		yy2 = Point(H1,yy.point[2])
	end
	kernel(xx1,yy1)*kernel(xx2,yy2)
end

# For kernel(Point,Point,..) calls (not usual I would think)
function kernel{T <: Tuple{Vararg{Point}}}(xx::T,yy::T)
	@assert length(xx)==length(yy)
	n=length(xx)
	res=1.0
	for i=1:n
		#might be able to use a fancier one-liner but this will do:
		res*=kernel(xx[i],yy[i])		
	end
	res
end

### Ground space (aka. sample space) for RKHS spaces


# Points in the ground space aka sample space
immutable Point{H <: RKHS,T}
	space::H
	point::T
	function call{H,T}(::Type{Point}, H1::H, x::T)
        @assert iselement(H1,x)
        new{H,T}(H1,x)
    end
end


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
