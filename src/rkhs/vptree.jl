##############
# BC 08/2016
# VP Tree
##############



# In some cases such as this package, I could dispatch directly on:
# distance{T}(x1::T,x2::T)
# and not carry the distance in the VPTree object.
# In the interest of generality and with a view towards plugging to Distances.jl, 
# I use `evaluate(distance::D,x1::T,x2::T)` to computes `d(x1,x2)`.
abstract Distance


# A (binary) tree representation of a vector (`data`) of points of length `n`
# is represented by a (2 x n) array (`array`) of indices such that
# `array[1,i]` and `array[2,i]` are the indices of the left and right children nodes
# of the node `i`.
# Thus you go down the three from `data[i]` to `data[array[1,i]]` and `data[array[2,i]]` (cheap)
# and up the tree from `data[i]` to `data[parent(i)]` where
# `parent(i)=ind2sub((2,n),find(x -> x==i,array)[1])[2]`   (expensive)    

immutable VPTree{T, D <: Distance}
    data::Vector{T}             # the data is a vector of points::T
    distance::D                 # `evaluate(distance::D,x1::T,x2::T)` computes `d(x1,x2)`
    array::Matrix{Int}          # array representation of the tree 
    root::Int                   # data[root] is the root of the tree
    radius::Vector{Float64}
end

#random splice
rpop!(v)=splice!(v,rand(1:length(v)))

splice_or_insert!(v,r,x)=isempty(r)?splice!(v,r,x):insert!(v,r.start,x)

function vp_node!(tree::VPTree,indices::AbstractVector{Int},parent::Int)
	n=length(indices)
	if n==1
		tree.array[1,parent]=indices[1]
		tree.radius[parent]=1.01*evaluate(tree.distance,tree.data[parent],tree.data[indices[1]])
	elseif n==2
		tree.array[1,parent]=indices[1]
		tree.array[2,parent]=indices[2]
		dist1=evaluate(tree.distance,tree.data[parent],tree.data[indices[1]])
		dist2=evaluate(tree.distance,tree.data[parent],tree.data[indices[2]])
		tree.radius[parent]=(dist1+dist2)/2
	elseif n>2 
		sorted_distances=Float64[]
		sorted_indices=Int[]
		for i=indices
			d_i=evaluate(tree.distance,tree.data[parent],tree.data[i])
			r_i=searchsorted(sorted_distances,d_i)
			splice_or_insert!(sorted_distances,r_i,d_i)
			splice_or_insert!(sorted_indices,r_i,i)
		end
		split_index=div(n,2)
		
		tree.radius[parent]=mean(sorted_distances[split_index:split_index+1])

		close_range=collect(1:split_index)
		close_root_index=rpop!(close_range)
		close_root=sorted_indices[close_root_index]
		close=view(sorted_indices,close_range)
		tree.array[1,parent]=close_root
		vp_node!(tree,close,close_root)
		
		far_range=collect(split_index+1:n)
		far_root_index=rpop!(far_range)
		far_root=sorted_indices[far_root_index]
		far=view(sorted_indices,far_range)
		tree.array[2,parent]=far_root
		vp_node!(tree,far,far_root)
	end
end

function VPTree(data::Vector,distance::Distance)
	n=length(data)
	indices=collect(1:n)
	root=rpop!(indices)
	tree=VPTree(data,distance,root,zeros(Int,2,n),zeros(n))
	vp_node!(tree,indices,root)
	tree
end


# returns the indices of the `k` nearest neighbors
function knn{T}(tree::VPTree{T}, x::T, k::Int)
	nodes_to_visit=[tree.root]
	tau=Inf
	neighbors=Int[]
	distances=Float64[]

	push_nz!(array,x)=((x==0)?nothing:push!(array,x))

	while length(nodes_to_visit) > 0
		
		node=pop!(nodes_to_visit)
		d=evaluate(tree.distance,tree.data[node],x)
		
		if d<tau #closest neighbor found so far
			r=searchsorted(distances,d)
			splice_or_insert!(distances,r,d)
			splice_or_insert!(neighbors,r,node)
			if length(distances) >= k
				tau=distances[k]
			end
		end

		#left child  = array[1,node] = close points 
		#right child  = array[2,node] = far points 
		if d < tree.radius[node]
			d  < tree.radius[node] + tau && push_nz!(nodes_to_visit,tree.array[1,node]) 
			d >= tree.radius[node] - tau && push_nz!(nodes_to_visit,tree.array[2,node])
		else
			d >= tree.radius[node] - tau && push_nz!(nodes_to_visit,tree.array[2,node])
			d  < tree.radius[node] + tau && push_nz!(nodes_to_visit,tree.array[1,node])
		end
	
	end
	# println(neighbors)
	neighbors[1:k]
end


