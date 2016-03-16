######################################
### TYPES AND CONSTRUCTORS
######################################


abstract RKHS

type RKHS2{H1<:RKHS,H2<:RKHS} <: RKHS end

type RKHSLeftElement{H <: RKHS,PT<:AbstractVector,WT<:AbstractMatrix}
	weights::WT   #line vector
	points::PT
	function RKHSLeftElement(weights,points)
		@assert length(weights)==length(points)
		new(weights,points)
	end
end

RKHSLeftElement{H<:RKHS,PT<:AbstractVector,WT<:AbstractMatrix}(::Type{H},weights::WT,points::PT)= RKHSLeftElement{H,PT,WT}(weights,points)

function RKHSLeftElement{H}(::Type{H},measure::Distribution,m::Int=100)
	points=rand(measure,m)
	weights=fill(1/m,1,m)
	RKHSLeftElement(H,weights,points)
end

type RKHSRightElement{H <: RKHS,PT<:AbstractMatrix,WT<:AbstractVector}
	points::PT   #line vector
	weights::WT
	function RKHSRightElement(weights,points)
		@assert length(weights)==length(points)
		new(points,weights)
	end
end

RKHSRightElement{H<:RKHS,PT,WT}(::Type{H},points::PT,weights::WT)= RKHSRightElement{H,PT,WT}(points,weights)


#make constructors checks that H1 and H2 are different
type RKHSMap{H1<: RKHS,H2<: RKHS,PT1<:AbstractMatrix,WT<:AbstractMatrix,PT2<:AbstractVector}
	leftpoints::PT1  #line vector
	weights::WT 
	rightpoints::PT2
	function RKHSMap(leftpoints,weights,rightpoints)
		@assert length(leftpoints)==size(weights,1)
		@assert length(rightpoints)==size(weights,2)
		new(leftpoints,weights,rightpoints)
	end
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





# I used the following to make it generic (with a view towards ForwardDiff-ing)
# But it created type inference issues
# function kernel{H<:RKHS}(::Type{H},xx1::AbstractVector,xx2::AbstractMatrix)
# 	kern(x1,x2)=kernel(H,x1,x2)
# 	ret=Array(typeof(kern(xx1[1],xx2[1])),length(xx1),length(xx2))  
# 	broadcast!(kern,ret,xx1,xx2)
# 	ret
# end




function kernel{H<:RKHS}(::Type{H},xx1::AbstractVector,xx2::AbstractMatrix)
	# ret=Array(typeof(kern(xx1[1],xx2[1])),length(xx1),length(xx2))  
	d1,d2=length(xx1),length(xx2)
	ret=Array(Float64,d1,d2)
	for i2=1:d2
		for i1=1:d1
			ret[i1,i2]=kernel(H,xx1[i1],xx2[i2])::Float64
		end
	end
	ret
end

# repeat of previous definition, only purpose: avoid ambiguities
function kernel{H1<:RKHS,H2<:RKHS}(::Type{RKHS2{H1,H2}},xx1::AbstractVector,xx2::AbstractMatrix)
	d1,d2=length(xx1),length(xx2)
	ret=Array(Float64,d1,d2)
	for i2=1:d2
		for i1=1:d1
			ret[i1,i2]=kernel(H1,xx1[i1][1],xx2[i2][1])*kernel(H2,xx1[i1][2],xx2[i2][2])::Float64
		end
	end
	ret
end

kernel{H1<:RKHS,H2<:RKHS}(::Type{RKHS2{H1,H2}},x1::Tuple,x2::Tuple)=kernel(H1,x1[1],x2[1])*kernel(H2,x1[2],x2[2])


function leftcompact{H1<:RKHS,H2<:RKHS}(joint::RKHSMap{H1,H2})
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
		#add the new line slice(joint.weights,i,:) to the correct line slice(cw,clpd[x],:)
		broadcast!(+,slice(cw,clpd[x],:),slice(cw,clpd[x],:),slice(joint.weights,i,:))
	end
	RKHSMap(H1,H2,slice(clp,:,1:index),slice(cw,1:index,:),joint.rightpoints)
end



function rightcompact{H1<:RKHS,H2<:RKHS}(joint::RKHSMap{H1,H2})
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
		#add the new column slice(joint.weights,:,j) to the correct column slice(cw,:,crpd[x])
		broadcast!(+,slice(cw,:,crpd[x]),slice(cw,:,crpd[x]),slice(joint.weights,:,j))
	end
	RKHSMap(H1,H2,joint.leftpoints,slice(cw,:,1:index),slice(crp,1:index))
end

compact{H1<:RKHS,H2<:RKHS}(joint::RKHSMap{H1,H2})=leftcompact(rightcompact(joint))


function norm{H1 <: RKHS}(measure::RKHSLeftElement{H1})
	G=kernel(H1,measure.points,line(measure.points))
	sqrt((measure.weights*G*vec(measure.weights))[1])
end

# most time is spent in computing the Gramian matrix in a naive implementation
# this is clearly not necessary
function distance(G11,weights1,weights2)
	dw=weights1-weights2
	sqrt((dw'*G11*dw)[1])
end

dist(G1,weights1,B1,weights2,B2)=distance(G1,weights1,proj(weights2,B2,B1))
dist(G1,weights1,B1,distribution,n=1000)=distance(G1,weights1,proj(fill(1/n,n),rand(distribution,n),B1))

function distance{H1 <: RKHS}(measure1::RKHSLeftElement{H1},measure2::RKHSLeftElement{H1})
	G11=kernel(H1,measure1.points,line(measure1.points))
	G12=kernel(H1,measure1.points,line(measure2.points))
	G22=kernel(H1,measure2.points,line(measure2.points))
	s11=measure1.weights*G11*vec(measure1.weights)
	s22=measure2.weights*G22*vec(measure2.weights)
	s12=measure1.weights*G12*vec(measure2.weights)
	sqrt(s11[1]+s22[1]-2*s12[1])
end

distance{H1 <: RKHS}(measure1::RKHSLeftElement{H1},measure2::Distribution,m=100)=distance(measure1,RKHSLeftElement(H1,measure2,m))
distance{H1 <: RKHS}(measure2::Distribution,measure1::RKHSLeftElement{H1},m=100)=distance(measure1,measure2)

moment(measure::RKHSLeftElement,n::Int)=dot(vec(measure.weights),measure.points.^n)
