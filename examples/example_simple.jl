reload("HiddenMarkovModels")

module dev

using Plots
using Base.Test
using HiddenMarkovModels
using Distributions




function histo(dd,m=50)
	bins=linspace(-5,5,m)
	delta=bins[2]-bins[1]
	counts=zeros(m)
	for i=1:m
		indices=(bins[i]-delta/2 .<= dd.points .< bins[i]+delta/2)
		counts[i]=sum(dd.weights[indices])
	end
	bins,counts
end

function comp(N=100,lambda=0.0,a=1,var=1+0.2^2)
	xx1=5*randn(N)
	xx2=zeros(N)
	for i=1:N
		xx2[i]=rand(Normal(a*xx1[i],sqrt(var)))
	end

	mu(x)=a*x

	Hx=HG{1}
	Hy=HG{2}
	marg=RKHSLeftElement(HG{1},Normal(0,1))
	joint=RKHSMap(Hx,Hy,line(xx1),eye(N),xx2)
	mk1=conditioningrule(joint,lambda=lambda)
	# cond=sumrule(Dirac(Hx,x1),mk1)

	# println(distance(cond,true_cond))
	muhat(x)=moment(sumrule(Dirac(Hx,x),mk1),1)
	# s2hat=moment(cond,2)-muhat^2
	
	xx=linspace(-2,2,30)
	plot(xx,map(muhat,xx))
	plot!(xx,map(mu,xx))

	# println("true mean = $(round(mu(x1),3)), approx = $(round(muhat(x1),3))")
	# println("true sd = $b, approx = $(round(sqrt(s2hat),3))")

	# bins,counts=histo(marg)
	# bins,counts=histo(cond)
	# scatter(bins,counts)
	# dd1,dd2,joint
end

comp(1000,10)



# fil,filt=comparefilters(2)


# using ProfileView
# Profile.clear()
# @profile fil,filt=compare(10);
# ProfileView.view()







# @time pre,up=filtr(mk,ini,data);



end