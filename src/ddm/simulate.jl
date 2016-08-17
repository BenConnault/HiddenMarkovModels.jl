#sample (x,y) from a matrix of joint probabilities.
function wsample2(mu)
	dx,dy=size(mu)
	ind2sub((dx,dy),wsample(1:dx*dy,vec(mu)))
end



function rand(model::DynamicDiscreteModel,T::Int)
	#throw error if not calibrated
	data=Array(Int,T)
	dx,dy=size(model.mu)
	x,data[1]=wsample2(model.mu)
	for t=2:T
		x,data[t]=wsample2(reshape(model.m[x,data[t-1],:,:],(dx,dy)))
	end
	data
end


#simulate iid individuals with heterogeneous number of periods of observation
function rand(model::DynamicDiscreteModel,T::Array{Int,1})
	n=length(T) 
	data=Array(Array,n)
	for i=1:n
		data[i]=rand(model,T[i])
	end
	data
end

# (T,n) -> n iid individuals with T periods of observation each
rand(model::DynamicDiscreteModel,T,n)=rand(model,fill(T,n))
# rand(model::StatisticalModel,Tn::Tuple{Int,Int})=rand(model,Tn[1],Tn[2])

