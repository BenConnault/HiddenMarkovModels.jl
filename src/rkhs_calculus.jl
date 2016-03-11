######################################
### SUM RULES: MEASURE x FUNCTION ####
######################################

# Taking plain expectation
sumrule{H}(measure::RKHSLeftElement{H},func::RKHSRightElement{H})=measure.weights*kernel(H,measure.points,func.points)*func.weights

###################################
### SUM RULES: MEASURE x TRANSITION ##########
###################################

# plain Markov kernel transition
function sumrule{H1,H2}(measure::RKHSLeftElement{H1},transition::RKHSMap{H1,H2})
	G=kernel(H1,measure.points,transition.leftpoints)
	RKHSLeftElement(H2,measure.weights*G*transition.weights,transition.rightpoints)
end

# partial Markov kernel transition:
# P(dx1) lambda((x1,x2),dx3)
# particular case of interest: P(dx1) is a Dirac, this gives lambda((x1=some_x1,x2),dx3)
function sumrule{H1,H2,H3}(measure::RKHSLeftElement{H1},transition::RKHSMap{RKHS2{H1,H2},H3})
	leftpoints1,leftpoints2=unpack(transition.leftpoints)
	G=kernel(H1,measure.points,line(leftpoints1))
	V=vec(measure.weights*G)
	RKHSMap(H2,H3,line(leftpoints2),scale(V,transition.weights),transition.rightpoints)
end

sumrule{H1,H3}(measure::RKHSLeftElement{H1},transition::RKHSMap{RKHS2{H1,H1},H3})=error("Ambiguous sumrule.")

# partial Markov kernel transition:
# P(dx2) lambda((x1,x2),dx3)
# particular case of interest: P(dx2) is a Dirac, this gives lambda((x1,x2=some_x2),dx3)
function sumrule{H1,H2,H3}(measure::RKHSLeftElement{H2},transition::RKHSMap{RKHS2{H1,H2},H3})
	leftpoints1,leftpoints2=unpack(transition.leftpoints)
	G=kernel(H2,measure.points,line(leftpoints2))
	V=vec(measure.weights*G)
	RKHSMap(H1,H3,line(leftpoints1),scale(V,transition.weights),transition.rightpoints)
end

# example: partial of Q((x,y),(dx',dy'))=Q(x,(dx',dy')) with respect to P(dy)=Dirac(y_value) => Q(x,(dx',dy'))
sumrule{H1,H2,H3}(measure::RKHSLeftElement{H1},transition::RKHSMap{H2,H3})=transition

# Markov kernel transition with partial conditional independence on one of the inputs:
# P((dx1,dx1)) lambda((x1,x2),dx3)=P((dx1,dx1)) lambda(x1,dx3)
# useful for strict hidden Markov models
# might use a "compact" keyword
sumrule{H1,H3}(measure::RKHSLeftElement{RKHS2{H1,H1}},transition::RKHSMap{H1,H3})=
	error("Ambiguous sumrule: make distinct copies of isomorphic spaces")
sumrule{H1,H2,H3}(measure::RKHSLeftElement{RKHS2{H1,H2}},transition::RKHSMap{H1,H3})=sumrule(marginal(H1,measure),transition)
sumrule{H1,H2,H3}(measure::RKHSLeftElement{RKHS2{H1,H2}},transition::RKHSMap{H2,H3})=sumrule(marginal(H2,measure),transition)


###################################
### TRANSITION x FUNCTION #########
###################################

#plain g(x1)=integral_x2 P(x1,dx2) f(x2)
function sumrule{H1,H2}(transition::RKHSMap{H1,H2},func::RKHSRightElement{H2})
	G=kernel(H2,transition.rightpoints,func.points)
	RKHSRighElement(H1,transition.leftpoints,transition.weights*G*func.weights)
end

#aa is an array (think line or vector) of tuples
function unpack(aa)
	n=length(aa)
	a1=Array(eltype(aa[1][1]),n)
	a2=Array(eltype(aa[1][2]),n)
	for i=1:length(aa)
		a1[i]=aa[i][1]
		a2[i]=aa[i][2]
	end
	a1,a2
end

#outlambda(x1,dx3)=integral_x2 lambda(x1,(dx2,dx3)) f(x2)
function sumrule{H1,H2,H3}(transition::RKHSMap{H1,RKHS2{H2,H3}},func::RKHSRightElement{H2})
	rightpoints2,rightpoints3=unpack(transition.rightpoints)
	D=kernel(H2,rightpoints2,vec(func.points))
	RKHSMap(H1,H3,transition.leftpoints,scale(transition.weights,D),rightpoints3)
end

sumrule{H1,H2}(transition::RKHSMap{H1,RKHS2{H2,H2}},func::RKHSRightElement{H2})=error("Ambiguous sumrule.")

#outlambda(x1,dx2)=integral_x3 lambda(x1,(dx2,dx3)) f(x3)
function sumrule{H1,H2,H3}(transition::RKHSMap{H1,RKHS2{H2,H3}},func::RKHSRightElement{H3})
	rightpoints2,rightpoints3=unpack(transition.rightpoints)
	D=kernel(H3,rightpoints3,vec(func.points))
	RKHSMap(H1,H2,transition.leftpoints,scale(transition.weights,D),rightpoints2)
end



###################################
### TRANSITION x TRANSITION #######
###################################

# plain Markov kernel transition composition:
# lambda3(x1,dx3)= integral_dx2 lambda1(x1,dx2) lambda2(x2,dx3)
function sumrule{H1,H2,H3}(transition1::RKHSMap{H1,H2},transition2::RKHSMap{H2,H3})
	G=kernel(H2,transition1.rightpoints,transition2.leftpoints)
	RKHSMap(H1,H3,transition1.leftpoints,transition1.weights*G*transition2.weights,transition2.rightpoints)
end




# Tedious to implement all cases. 
# In particular because seemingly-more-general cases cannot be used for less general dependency structures because 
# the (x,w,y) representation cannot carry conditional independences like P(y|x)=P(y)

# function sumrule{Hx1,Hy1,Hx2,Hy2}(transition::RKHSMap{Tuple{Hx1,Hy1},Hx2},transition::RKHSMap{Tuple{Hx2,Hy1},Hy2})

# 	RKHSMap{Tuple{Hx1,Hy1},Tuple{Hx2,Hy2}}()
# end

# function sumrule{Hx1,Hy1,Hx2,Hy2}(transition::RKHSMap{Tuple{Hx1,Hy1},Hy2},transition::RKHSMap{Tuple{Hx1,Hy2},Hx2})
# 	G=kernel
# 	RKHSMap{Tuple{Hx1,Hy1},Tuple{Hx2,Hy2}}()
# end

# measure * transition * function
function sumrule{H1,H2}(measure::RKHSLeftElement{H1},transition::RKHSMap{H1,H2},func::RKHSRightElement{H2})
	sumrule(sumrule(measure,transition),func)
end


###################################
### CHAIN RULES ###################
###################################

# Q((dx1,dx2))=P(dx1)lambda(x1,dx2)
# returns a joint distribution as a map
# similar to a sumrule but we don't contract on the first index
function chainrule{H1,H2}(measure::RKHSLeftElement{H1},transition::RKHSMap{H1,H2},G1::AbstractMatrix)
	weights=scale(vec(measure.weights),G1*transition.weights)
	RKHSMap(H1,H2,line(measure.points),weights,transition.rightpoints)
end

function transposechainrule{H1,H2,WT <: AbstractMatrix,PT1a,PT1b,PT2}(measure::RKHSLeftElement{H1,PT1a,WT},transition::RKHSMap{H1,H2,PT1b,WT,PT2},G1::WT)
	weights=scale(At_mul_Bt(transition.weights,G1),slice(measure.weights,1,:))
	# RKHSMap(H2,H1,line(transition.rightpoints),weights,measure.points)
	RKHSMap(H2,H1,transition.rightpoints',weights,measure.points)    #seems faster than line() when profiling
end


chainrule{H1,H2}(measure::RKHSLeftElement{H1},transition::RKHSMap{H1,H2})=
	chainrule(measure,transition,kernel(H1,measure.points,transition.leftpoints))

#degenerate chain rule for independent distributions
function chainrule{H1,H2}(measure1::RKHSLeftElement{H1},measure2::RKHSLeftElement{H2})
	weights=broadcast(*,vec(measure1.weights),measure2.weights)
	RKHSMap(H1,H2,reshape(measure1.points,1,length(measure1.points)),weights,measure2.points)
end

# Q(x1,(dx2,dx3))=P(x1,dx2)lambda(x2,dx3)
function chainrule{H1,H2,H3}(transition1::RKHSMap{H1,H2},transition2::RKHSMap{H2,H3})
	G=kernel(H1,transition1.rightpoints,transition2.leftpoints)
	d1,d2a=size(transition1.weights)
	d2b,d3=size(transition2.weights)
	weights=similar(transition1.weights,d1,d2a,d3)
	for i1=1:d1
		weights[i1,:,:]=scale(slice(transition1.weights,i1,:),G*transition2.weights)
	end
	rightpoints=[ (transition1.rightpoints[i2],transition2.rightpoints[i3]) for i2=1:d2a,i3=1:d3 ]
	RKHSMap(H1,RKHS2{H2,H3},transition1.leftpoints,reshape(weights,d1,d2a*d3),vec(rightpoints))
end

###################################
### CONDITIONING RULE
###################################

# from C_XY = (x1 W y), do:
# C_XX = (x1 D x1), D = diagm(sum(W,2))
# C_XX \ C_XY  = (x1 D x1) \ (x2 W y) \approx (x1 [D^-1 G(x1,x2)^-1 W] y)  

# Comments:
# (1) the expression below is definitely valid for the particular case of interest:
#   (1a) discrete distributions
#   (1b) sample-based embedding
#     I am not sure if it is valid in full generality. Might want to try by giving explicit densities
#     on a grid
# (2) computation might be somewhat wasteful in the frequent case where joint.weights is diagonal 
# (3) I could specialize on WT <: Diagonal or sparse or other(weightype)

# condition on H1 always
# transpose before using if you want to condition on H2

#########

function conditioningrule_inv{H1,H2}(joint::RKHSMap{H1,H2},Gi::AbstractMatrix;lambda=0.0)
	D=1./vec(sum(full(joint.weights),2))    #full() is a stopgap unti sum(m,2) works for ::Diagonal matrices
	println("positive D's: ",sum(D.>0)," / ",length(D))
	weights=Gi*full(joint.weights)  #full() is a stopgap unti a\m works for m::Diagonal matrices
	weights=scale!(D,weights)
	
	# weights=(scale(G,D)+lambda*I)\full(joint.weights)
	# println()
	# println(round(weights,3))
	RKHSMap(H1,H2,joint.leftpoints,weights,joint.rightpoints)
end

function conditioningrule_inv{H1,H2}(joint::RKHSMap{H1,H2};lambda=0.0)
	G1=kernel(H1,vec(joint.leftpoints),joint.leftpoints)
	G1i=inv(Symmetric(G1))
	G1i=(G1i.*(abs(G1i).>.01))
	conditioningrule(joint,G1i,lambda=lambda)
end
##########

#this is feeding the non-inverted Gramian
function conditioningrule{H1,H2}(joint::RKHSMap{H1,H2},G::AbstractMatrix;lambda=0.0)
	D=vec(sum(full(joint.weights),2))    #full() is a stopgap unti sum(m,2) works for ::Diagonal matrices
	println("positive D's: ",sum(D.>0)," / ",length(D))
	M1=scale(G,D)
	# println("det=" ,det(G))
	# println(round(inv(G),3))
	M2=M1*M1
	for i=1:size(M2,1)
		M2[i,i]+=lambda
	end
	M3=M2\full(joint.weights)  #full() is a stopgap unti a\m works for m::Diagonal matrices
	# weights=G*M2
	# scale!(D,weights)
	#or for numerical stability: 
	weights=scale(D,G)*M3
	
	# weights=(scale(G,D)+lambda*I)\full(joint.weights)
	# println()
	# println(round(weights,3))
	RKHSMap(H1,H2,joint.leftpoints,weights,joint.rightpoints)
end

#wrapper for the previous one
conditioningrule{H1,H2}(joint::RKHSMap{H1,H2};lambda=0.0)=
	conditioningrule(joint,kernel(H1,vec(joint.leftpoints),joint.leftpoints),lambda=lambda)



function conditioningrule2{H1,H2}(joint::RKHSMap{H1,H2};lambda=0.0)
	D=vec(sum(full(joint.weights),2))    #full() is a stopgap unti sum(m,2) works for ::Diagonal matrices
	println("positive D's: ",sum(D.>0)," / ",length(D))
	G=kernel(H1,vec(joint.leftpoints),joint.leftpoints)
	scale!(G,D)
	# println("det=" ,det(G))
	# println(round(inv(G),3))
	weights=(G+lambda*I)\full(joint.weights)  #full() is a stopgap unti a\m works for m::Diagonal matrices
	# scale!(D,weights)
	# println()
	# println(round(weights,3))
	RKHSMap(H1,H2,joint.leftpoints,weights,joint.rightpoints)
end

conditioningrule{H1,H2}(joint::RKHSLeftElement{RKHS2{H1,H2}};lambda=0.0)=conditioningrule(RKHSMap(joint);lambda=0.0)


###################################
### BAYES RULE
###################################

bayesrule{H1,H2}(prior::RKHSLeftElement{H1},conditional::RKHSMap{H1,H2};lambda=0.0)=
	conditioningrule(transpose(chainrule(prior,conditional)),lambda=lambda)

bayesrule{H1,H2}(prior::RKHSLeftElement{H1},conditional::RKHSMap{H1,H2},G1::AbstractMatrix,G2i::AbstractMatrix;lambda=0.0)=
	conditioningrule(transposechainrule(prior,conditional,G1),G2i,lambda=lambda)

# kernel(T,leftpoints,rightpoints)=broadcast(kernel(T),measure.points,func.points)