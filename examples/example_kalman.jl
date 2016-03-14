reload("HiddenMarkovModels")

module dev

using Plots
using AverageShiftedHistograms
using Base.Test
using HiddenMarkovModels
using StateSpace
using Distributions
import Base.rand

# # from StateSpace.jl
# type LinearGaussianSSM{T} <: AbstractLinearGaussian
# 	F::Matrix{T}   	# process matrix
# 	V::Matrix{T}	# process variance
# 	G::Matrix{T}	# observation matrix
# 	W::Matrix{T}	# observation variance


function rand(model::LinearGaussianSSM,ini::AbstractVector,T::Int)
	dx,dy=size(model.G)
	@assert length(ini)==dx
	xx=zeros(T,dx)
	yy=zeros(T,dy)
	xx[1,:]=ini
	yy[1,:]=rand(MvNormal(model.G*slice(xx,1,:),model.W))
	for t=2:T
		xx[t,:]=rand(MvNormal(model.F*slice(xx,t-1,:),model.V))
		yy[t,:]=rand(MvNormal(model.G*slice(xx,t,:),model.W))
	end
	xx,yy
end

rand(model::LinearGaussianSSM,ini::Number,T::Int)=rand(model,[ini],T)


function rand(model::LinearGaussianSSM,T::Int)
	dx=size(model.F,1)
	ini=MvNormal(zeros(dx),model.V)
	rand(model,ini,T)
end


function toymodel()
	pm = 0.98     # process model parameter
	pc = 0.2^2 		# process variance
	om = 1.0     # observation model parameter
	oc = 1.0     # observation variance
	LinearGaussianSSM(pm, pc, om, oc) 
end


# P(Y_t+1|X_t=x1)  (=> mk1 in rkhs filtering)
mu1(model::LinearGaussianSSM,x1)=model.G*model.F*x1
sigma1(model::LinearGaussianSSM,x1)=model.G*model.V*model.G'+model.W
# P(X_t+1|X_t=x1,Y_{t+1}=y_2)  (=> mk2 in rkhs filtering)
mu2(model::LinearGaussianSSM,x1,y2)=model.F*x1-model.V*model.G'*((model.G*model.V*model.G'+model.W)\(y2-model.G*model.F*x1))
sigma2(model::LinearGaussianSSM,x1,y2)=model.V-model.V*model.G'*(model.G*model.V*model.G'+model.W)\model.G*model.V


function translate2RKHS(model::LinearGaussianSSM,N=100,lambda=0.0)
	
	xx1=1*randn(N)
	xx2=zeros(N)
	yy2=zeros(N)
	for i=1:N
		xx2[i]=rand(MvNormal(model.F*[xx1[i]],model.V))[1]
		yy2[i]=rand(MvNormal(model.G*[xx2[i]],model.W))[1]
	end

	Hx=HG{1}
	Hy=HG{2}
	joint=RKHSMap(Hx,Hy,line(xx1),eye(N),yy2)
	mk1=conditioningrule(joint,lambda=lambda)
	mk2=conditioningrule(RKHSMap(RKHS2{Hx,Hy},Hx,[(xx1[j],yy2[j]) for i=1,j=1:N],eye(N),xx2),lambda=lambda)

	# plot(ash(xx1,nbin=round(Int,sqrt(N))))
	# plot(ash(yy2,nbin=round(Int,sqrt(N))))
	
	mu1hat(x1)=round(moment(sumrule(Dirac(Hx,x1),mk1),1),4)
	function mu2hat(x1,y2)
		marginal=RKHSLeftElement(chainrule(Dirac(Hx,x1),Dirac(Hy,y2)))
		round(moment(sumrule(marginal,mk2),1),4)
	end

	# xx=linspace(-5,5,50)
	# plot(xx,map(mu1hat,xx))
	# plot!(xx,map(x->mu1(model,x)[1],xx))

	# for y2=0.0
	# 	plot(xx,map(x->mu2hat(x,y2),xx))
	# 	title!("y2=$y2")
	# 	plot!(xx,map(x->mu2(model,x,y2)[1],xx))
	# end

	# println(mu2(model,x10,y20)[1])
	# println(mu2hat(x10,y20))

	mk1,mk2,joint
end

function comparefilters(T=10,lambda2=1.0)
	# lambda=1.
	model=toymodel()	
	datx,daty=rand(model,0.0,T)
	
	println("x: ",round(datx,4)')
	println("y: ",round(daty,4)')

	Hx=HG{1}
	Hy=HG{2}
	initial=MvNormal(zeros(1),ones(1,1))
	fil=filter(model, daty',initial)

	N=1000
	lambda=1.0/sqrt(N)
	mk1,mk2,joint=translate2RKHS(model,N,lambda)
	ini=RKHSLeftElement(Hx,fill(1/N,1,N),vec(rand(fil.state[1],N)))

	function mu2hat(x1,y2)
		marginal=RKHSLeftElement(chainrule(Dirac(Hx,x1),Dirac(Hy,y2)))
		round(moment(sumrule(marginal,mk2),1),4)
	end
	# xx=linspace(-2,2,50)
	# y2=0.0
	# plot(xx,map(x->mu2hat(x,y2),xx))
	# title!("y2=$y2")
	# plot!(xx,map(x->mu2(model,x,y2)[1],xx))


	filt=filtr2(mk1,mk2,ini,daty,lambda=lambda2)
	
	true_filter=[fil.state[t].μ[1] for t=1:T]
	approx_filter=[moment(filt[t],1) for t=1:T]

	plot(1:T,true_filter)
	plot!(1:T,approx_filter)

	println("true filtered mean: ",round(fil.state[T].μ[1],4))
	println("approximate filtered mean: ",round(moment(filt[T],1),4))
	fil,filt
end

fil,filt=comparefilters(2,0.1);


# using ProfileView
# Profile.clear()
# @profile fil,filt=comparefilters(10);
# ProfileView.view()


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

function comparedist(x1=0)
	model=toymodel()
	mk1,mk2,initial,joint=translate2RKHS(model,100)
	Hx=HG{1}
	Hy=HG{2}
	mu=model.F[1]*model.G[1]*x1
	s2=(model.G[1]^2)*model.V[1]+model.W[1]
	dd1=Normal(mu,sqrt(s2))
	# println(round(mk1.leftpoints',3))
	# println(round(mk1.weights,3))
	dd2=sumrule(Dirac(Hx,0),mk1)

	println(distance(dd1,dd2))

	bins,counts=histo(dd2)
	scatter(bins,counts)
	dd1,dd2,joint
end

# dd1,dd2,joint=comparedist()







function ttest()
	A = [ 0.99 0.01 ;
	      0.01 0.99 ]
	B = [ 1/6   1/6   1/6   1/6   1/6   1/6 ;
	      1/10  1/10  1/10  1/10  1/10  1/2 ]
	p = [0.625, 0.375]

	data = [1, 2, 1, 6, 6] # observation sequence

	xx=[1,2]
	yy=[1,2,3,4,5,6]

	model=hmm((A,B),p*fill(1/6,1,6))
	fil,smo,cond=filtr(model,data)  # from DynamicDiscreteModels.jl

	Hx=HD{1}
	Hy=HD{2}

	mk1=RKHSMap(Hx,Hy,line(xx),A*B,yy)   # P(Y' | X)
	Q=[A[ix,jx]*B[jx,jy] for ix=1:2,jy=1:6,jx=1:2]
	Q=reshape(Q,12,2)
	Q=Q./sum(Q,2)
	mk2=RKHSMap(RKHS2{Hx,Hy},Hx,line([(ix,jy) for ix=xx,jy=yy]),Q,xx)   # P(X'| X,Y')
	ini=RKHSLeftElement(Hx,line(p),xx)

	filt=filtr2(mk1,mk2,ini,data)

	true_filter= [ 0.625   0.7332  0.8173  0.5884  0.3212 ;
	                 0.375   0.2668  0.1827  0.4116  0.6788 ]

	filter_from_rkhs=hcat(map(i->filt[i].weights[1],1:5),map(i->filt[i].weights[2],1:5))'
	filter_from_ddm=fil

	tol=1e-3

	@test norm(vec(filter_from_rkhs   - true_filter     ))   < tol 
	@test norm(vec(filter_from_rkhs   - filter_from_ddm ))   < tol 
end

# ttest()


# @time pre,up=filtr(mk,ini,data);



end