abstract Node

immutable InternalNode{LT <: Node, RT <: Node} <: Node
	index::Int     # index of current element in data list
	left::LT       # indices of all close elements
	right::RT      # indices of all far elements
end

immutable Leaf <: Node end

immutable VPTree{T, D <: Distance, RT <: Node}
    data::Vector{T}  
    distance::D      
    root::RT
end


function VPNode(data,distance,indices)
	n=length(indices)
	if n==0
		Leaf()
	elseif n==1
		InternalNode(indices[1],Leaf(),Leaf())
	elseif n==2
		InternalNode(indices[1],InternalNode(indices[2],Leaf(),Leaf()),Leaf())
	elseif n==3
		InternalNode(indices[1],InternalNode(indices[2],Leaf(),Leaf()),InternalNode(indices[3],Leaf(),Leaf()))
	else 
		node_index=rand(indices)
		sorted_distances=Float64[]
		sorted_indices=Int[]
		for i=1:n
			if i != node_index
				d_i=evaluate(distance,data[node_index],data[i])
				r_i=searchsorted(sorted_distances,d_i)
				splice!(sorted_distances,r_i,d_i)
				splice!(sorted_indices,r_i,i)
			end
		end
		split_index=div(n,2)
		close=slice(sorted_indices,1:split_index)
		far=slice(sorted_indices,split_index+1:n-1)
		InternalNode(node_index,VPNode(data,distance,close),VPNode(data,distance,far))
	end
end

VPTree(data,distance)=VPTree(data,distance,VPNode(data,distance,collect(1:length(data))))



function knn{T}(tree::VPTree{T}, x::T, k::Int)
	nodes_to_visit=




# function searchtree(Hx,Hy,Bxy)
# 	dx,dy=length(Bxy[1][1]),length(Bxy[1][2])
# 	bxy=repack(Bxy)
# 	# BruteTree(bxy, XYMetric{Hx,Hy}((dx,dy)); reorder = false)
# 	BallTree(bxy, XYMetric{Hx,Hy}((dx,dy)); reorder = false)
# end