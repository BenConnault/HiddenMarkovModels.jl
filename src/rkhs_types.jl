######################################
### TYPES AND CONSTRUCTORS
######################################


abstract RKHS

type RKHS2{H1<:RKHS,H2<:RKHS} <: RKHS end

type RKHSLeftElement{H <: RKHS,PT<:AbstractVector,WT<:AbstractMatrix}
	weights::WT   #line vector
	points::PT
end

RKHSLeftElement{H<:RKHS,PT,WT}(::Type{H},weights::WT,points::PT)= RKHSLeftElement{H,PT,WT}(weights,points)

type RKHSRightElement{H <: RKHS,PT<:AbstractMatrix,WT<:AbstractVector}
	points::PT   #line vector
	weights::WT 
end

RKHSRightElement{H<:RKHS,PT,WT}(::Type{H},points::PT,weights::WT)= RKHSRightElement{H,PT,WT}(points,weights)


#make constructors checks that H1 and H2 are different
type RKHSMap{H1<: RKHS,H2<: RKHS,PT1<:AbstractMatrix,WT<:AbstractMatrix,PT2<:AbstractVector}
	leftpoints::PT1  #line vector
	weights::WT 
	rightpoints::PT2
end

RKHSMap{H1<:RKHS,H2<:RKHS,PT1,WT,PT2}(::Type{H1},::Type{H2},leftpoints::PT1,weights::WT,rightpoints::PT2)=
	RKHSMap{H1,H2,PT1,WT,PT2}(leftpoints,weights,rightpoints)



######################################
### UTILITY FUNCTIONS
######################################


# go from a matrix to a vector representation for a joint distribution
# similar to "vec()" (and could be called that)
function RKHSLeftElement{H1<:RKHS,H2<:RKHS}(joint::RKHSMap{H1,H2})
	points=[(x,y) for x = joint.leftpoints,y=joint.rightpoints]
	RKHSLeftElement(RKHS2{H1,H2},vec(joint.weights)',vec(points))   #I want a copy of joint.weights here
end

# go from a vector to a matrix representation for a joint distribution
# could make a keyword argument for automatic compact
function RKHSMap{H1<:RKHS,H2<:RKHS}(joint::RKHSLeftElement{RKHS2{H1,H2}})
	leftpoints,rightpoints=unpack(joint.points)
	RKHSMap(H1,H2,line(leftpoints),Diagonal(vec(joint.weights)),rightpoints)
end


marginal{H1<:RKHS}(::Type{H1},joint::RKHSLeftElement{RKHS2{H1,H1}})=error("Ambiguous marginal.")


# could make a keyword argument for automatic compact
function marginal{H1<:RKHS,H2<:RKHS}(::Type{H1},joint::RKHSLeftElement{RKHS2{H1,H2}})
	leftpoints,rightpoints=unpack(joint.points)
	RKHSLeftElement(H1,joint.weights,leftpoints)
end

# could make a keyword argument for automatic compact
function marginal{H1<:RKHS,H2<:RKHS}(::Type{H2},joint::RKHSLeftElement{RKHS2{H1,H2}})
	leftpoints,rightpoints=unpack(joint.points)
	RKHSLeftElement(H2,joint.weights,rightpoints)
end

# could make a keyword argument for automatic compact
function marginals{H1<:RKHS,H2<:RKHS}(joint::RKHSLeftElement{RKHS2{H1,H2}})
	leftpoints,rightpoints=unpack(joint.points)
	RKHSLeftElement(H1,joint.weights,leftpoints),RKHSLeftElement(H2,joint.weights,rightpoints)
end

transpose{H1<:RKHS,H2<:RKHS}(joint::RKHSMap{H1,H2})=
	RKHSMap(H2,H1,line(joint.rightpoints),joint.weights',vec(joint.leftpoints))


Dirac{H<:RKHS}(::Type{H},x)=RKHSLeftElement(H,fill(1.0,1,1),[x])





function kernel{H<:RKHS}(::Type{H},xx1::AbstractVector,xx2::AbstractMatrix)
	kern(x1,x2)=kernel(H,x1,x2)
	broadcast(kern,xx1,xx2)
end

# repeat of previous definition, only purpose: avoid ambiguities
function kernel{H1<:RKHS,H2<:RKHS}(::Type{RKHS2{H1,H2}},xx1::AbstractVector,xx2::AbstractMatrix)
	kern(x1,x2)=kernel(RKHS2{H1,H2},x1,x2)
	## the following fails because of a bad type inference:
	# broadcast(kern,xx1,xx2)
	## we help the type inference explicitly:
	ret=Array(typeof(kern(xx1[1],xx2[1])),length(xx1),length(xx2))   
	broadcast!(kern,ret,xx1,xx2)
	ret
end

kernel{H1<:RKHS,H2<:RKHS}(::Type{RKHS2{H1,H2}},x1,x2)=kernel(H1,x1[1],x2[1])*kernel(H2,x1[2],x2[2])

######################################
### CANONICAL DISCRETE RKHS
######################################

immutable DiscreteRKHS{T} <: RKHS end

typealias HD DiscreteRKHS

kernel{T<:DiscreteRKHS}(::Type{T},x1::Int,x2::Int)=1.0*(x1==x2)


function leftcompact{H1<:DiscreteRKHS,H2<:RKHS}(joint::RKHSMap{H1,H2})
	index=1
	cw=zeros(full(joint.weights))     			#cw  = compact weights
	clp=zeros(joint.leftpoints) 		        #clp = compact left points
	clpd=Dict(joint.leftpoints[1]=>index) 		#clpd = compact left points dictionnary
	clp[index]=joint.leftpoints[1]
	for (i,x) in enumerate(joint.leftpoints)
		if !haskey(clpd,x)
			index+=1
			clpd[x]=index
			clp[index]=x
		end		
		broadcast!(+,slice(cw,clp[x],:),slice(cw,clp[x],:),slice(joint.weights,i,:))
	end
	RKHSMap(H1,H2,slice(clp,:,1:index),slice(cw,1:index,:),joint.rightpoints)
end



function rightcompact{H1<:RKHS,H2<:DiscreteRKHS}(joint::RKHSMap{H1,H2})
	index=1
	cw=zeros(full(joint.weights))           	#cw  = compact weights
	crp=zeros(joint.rightpoints) 		        #crp = compact right points
	crpd=Dict(joint.rightpoints[1]=>index) 		#crpd = compact right points dictionnary
	crp[index]=joint.rightpoints[1]
	for (j,x) in enumerate(joint.rightpoints)
		if !haskey(crpd,x)
			index+=1
			crpd[x]=index
			crp[index]=x
		end		
		broadcast!(+,slice(cw,:,crp[x]),slice(cw,:,crp[x]),slice(joint.weights,:,j))
	end
	RKHSMap(H1,H2,joint.leftpoints,slice(cw,:,1:index),slice(crp,1:index))
end

compact{H1<:DiscreteRKHS,H2<:DiscreteRKHS}(joint::RKHSMap{H1,H2})=leftcompact(rightcompact(joint))
compact{H1<:RKHS,H2<:DiscreteRKHS}(joint::RKHSMap{H1,H2})=rightcompact(joint)
compact{H1<:DiscreteRKHS,H2<:RKHS}(joint::RKHSMap{H1,H2})=leftcompact(joint)
compact{H1<:RKHS,H2<:RKHS}(joint::RKHSMap{H1,H2})=joint
