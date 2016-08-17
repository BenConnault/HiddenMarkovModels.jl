#could be written more efficiently
#but is not meant to be evaluated repeatedly like the likelihood
#so not a priority


function viterbi(model::DynamicDiscreteModel,data::Array{Int,1})
	T=length(data)
	dx=size(model.mu,1)

	memory=Array(Int,dx,T-1)

	model.psi[:]=model.mu[:,data[1]]/sum(model.mu[:,data[1]])
	for t=2:T
		mm=reshape(model.m[:,data[t-1],:,data[t]],(dx,dx))
		for jx=1:dx
			model.psi[jx],memory[jx,t-1]=findmax(model.psi.*mm[:,jx])
		end
		model.psi[:]=model.psi/sum(model.psi)
	end
	viterbipath=Array(Int,T)
	viterbipath[T]=indmax(model.psi)
	for t=T-1:-1:1
		viterbipath[t]=memory[viterbipath[t+1],t]
	end
	viterbipath
end

#wrapper for panel data
function viterbi(model::DynamicDiscreteModel,data::Array{Array,1})
	n=length(data)
	viterbipaths=Array(Array{Int,1},n)
	for i=1:n
		viterbipaths[i]=viterbi(model,data[i])
	end
	viterbipaths
end