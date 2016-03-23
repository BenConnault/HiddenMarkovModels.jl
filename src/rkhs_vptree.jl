# In special cases such as this package, I could dispatch directly on:
# distance{T}(x::T,y::T)
# and not carry the distance in the VPTree object.
# In the interest of generality and with a view towards plugging to Distances.jl, 
# I don't follow this road.

immutable VPTree{T, D <: Distance}
    data::Vector{T}  
    distance::D      
    root::Int
    array::Matrix{Int}
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
		close=slice(sorted_indices,close_range)
		tree.array[1,parent]=close_root
		vp_node!(tree,close,close_root)
		
		far_range=collect(split_index+1:n)
		far_root_index=rpop!(far_range)
		far_root=sorted_indices[far_root_index]
		far=slice(sorted_indices,far_range)
		tree.array[2,parent]=far_root
		vp_node!(tree,far,far_root)
	end
end

function VPTree(data,distance)
	n=length(data)
	indices=collect(1:n)
	root=rpop!(indices)
	tree=VPTree(data,distance,root,zeros(Int,2,n),zeros(n))
	vp_node!(tree,indices,root)
	tree
end



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
	println(neighbors)
	neighbors[1:k]
end


