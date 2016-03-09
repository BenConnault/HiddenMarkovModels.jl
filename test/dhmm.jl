a=[.4 .6; .3 .7]
b=[.3 .1 .6; .5 .2 .3]
data=[1,2,3]
model=hmm((a,b))

function naiveprob(data)
	filter=transpose(model.mu[:,data[1]])
	for t=2:length(data)
		filter=sum(filter*slice(model.m,:,data[t-1],:,data[t]),1)
	end
	sum(filter)
end

@test vec(sum(model.m,(3,4)))≈fill(1,6) 
@test reshape(sum(model.m[:,1,:,:],4),2,2)≈a
@test loglikelihood(model,data)≈log(naiveprob(data))/length(data)
data=Array(Array,2)
data[1]=[1,2,3]
data[2]=[3,2,1]
@test loglikelihood(model,data)≈(log(naiveprob(data[1]))/length(data[1])+log(naiveprob(data[2]))/length(data[2]))/2

viterbi(model,data)

data=rand(model,10000)
abhat=em(model,data)

@test vecnorm(abhat[1]-a)<5e-2
@test vecnorm(abhat[2]-b)<5e-2