# project v=(w1,B1) on B2 by finding k nearest neighbours of each element 
# of B1 in B2, using pre-computed search tree representation of B2 `tree`.
# G2=B2'B2 is pre-computed\
# argmin_w2 |B1 w1-B2 w2| = < B1 w1-B2 w2 , B1 w1-B2 w2 > = 
# w1'B1'B1w1 - 2 w1' G12 w2 + w2' G2 w2 

#######################################################################################

# There is a design choice to be made between:
#    project(v::RKHSVector{H},B::RKHSBasis{H}) means whether
# (1) project on the full basis, even if the linear program can be prohibitively expensive
#     when the basis has many elements
# (2) project on k nearest neighbors where k is a reasonable default value like
#     k = 2*dimension(rkhs(H))

# alternative (1)
# project{H}(v::RKHSVector{H},B::RKHSBasis{H})=project(v,B,gramian(B,B))

# alternative (2)
# project{H}(v::RKHSVector{H},B::RKHSBasis{H})=project(v,RKHSBasisTree(B),2*dimension(rkhs(v.basis)))

#######################################################################################

#with a basistree argument there is no ambiguity between full or nearest neighbor projection
function project{H}(v::RKHSVector{H},basistree::RKHSBasisTree{H},k::Int=2*dimension(rkhs(v.basis)))
	tree,G2=basistree.tree,basistree.gram
	# @assert rkhs(v)==rkhs(tree) 
	B2points=tree.data
	w=zeros(length(B2points))
	aa=zeros(k)
	for i=1:length(v.basis)
		inns = knn(tree, v.basis.points[i], k)    #indices of nearest neighbors
		Gi=G2[inns,inns]
		G12=vec(gramian(rkhs(H),view(B2points,inns),[v.basis.points[i]]))   #watch out for type instability here
		w[inns]+=v.weights[i]*_project(Gi,G12)
	end
	w
end

project{H}(v::RKHSVector{H},B::RKHSBasis{H},k::Int)=project(v,RKHSBasisTree(B),k)



function project{H}(v::RKHSVector{H},B::RKHSBasis{H},G::Matrix{Float64})
	G12=vec(At_mul_B(v.weights,gramian(v.basis,B)))
	_project(G,G12)
end



function _project(G1::Matrix{Float64},G12::Vector{Float64})
	n=length(G12)
	m = Model(solver=IpoptSolver(print_level=0))
	@variable(m, 0.0 <= w2[1:n] <= 1.0 )
	# @defVar(m, 0.0 <= w2[1:k]  )
	@objective(m, Min,  (w2'*G1*w2)[1]-2*dot(G12,w2) )
	@constraint(m, sum(w2) == 1 )
	status = solve(m)
	getvalue(w2)
end

