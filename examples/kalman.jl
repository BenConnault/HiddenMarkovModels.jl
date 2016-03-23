reload("HiddenMarkovModels")

module dev

using Plots
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


mu1(model::LinearGaussianSSM,x1)=model.F*x1
mu2(model::LinearGaussianSSM,x2)=model.G*x2

#computes a markov kernel by doing a global projection of 
# the joint (1/n,B1), where B1 is a list of pairs (x_i,y_i)
# to a joint on (Bx x By),
# then conditioning
function markov(Hx,Hy,B1,Bx,By)
	dx,dy,n=length(Bx),length(By),length(B1)
	B2=vec([(x,y) for x=Bx,y=By])
	lB2=line(B2)
	G12=kernel(RKHS2{Hx,Hy},B1,lB2)	
	G2=kernel(RKHS2{Hx,Hy},B2,lB2)
	joint=reshape(project(fill(1.0/n,n),G12,G2),dx,dy)
	mk=joint./sum(joint,2)
	RKHSMap(Hx,Hy,line(Bx),mk,By)
end




function markov_nn(Hx,Hy,B1,Bx,By)
	dBx,dBy,n=length(Bx),length(By),length(B1)
	Hxy=RKHS2{Hx,Hy}
	dx,dy=length(Bx[1]),length(By[1])
	kxy=2*(dx+dy)
	B2=vec([(x,y) for x=Bx,y=By])
	G2=kernel(Hxy,B2,line(B2))
	tree=searchtree(Hx,Hy,B2)
	joint=reshape(proj_nn(Hxy,fill(1.0/n,n),B1,B2,tree,G2,kxy),dBx,dBy)
	mk=joint./sum(joint,2)
	RKHSMap(Hx,Hy,line(Bx),mk,By)
end


function strict_hmm(model::LinearGaussianSSM,n=20,N=500)
	
	xx1=2*randn(N)
	xx2=zeros(N)
	yy2=zeros(N)
	for i=1:N
		xx2[i]=rand(MvNormal(model.F*[xx1[i]],model.V))[1]
		yy2[i]=rand(MvNormal(model.G*[xx2[i]],model.W))[1]
	end

	Hx=HG{1}
	Hy=HG{2}
		
	Bx=sort(2*randn(n))
	By=sort(2*randn(n))

	B1=[(xx1[i],xx2[i]) for i=1:N]
	mk1=markov_nn(Hx,Hx,B1,Bx,Bx)   # P(X_{t+1}|X_t) expressed in (Bx -> Bx)  
	B2=[(xx2[i],yy2[i]) for i=1:N]
	mk2=markov_nn(Hx,Hy,B2,Bx,By)   # P(Y_{t+1}|X_{t+1}) expressed in (Bx -> By)  

	
	mu1hat(x1)=(proj_nn([1.0],[x1],Bx)'*mk1.weights*Bx)[1]
	mu2hat(x2)=(proj_nn([1.0],[x2],Bx)'*mk2.weights*By)[1]
	

	xx=linspace(-5,5,50)
	plot(xx,map(mu1hat,xx))
	plot!(xx,map(x->mu1(model,x)[1],xx))

	plot(xx,map(mu2hat,xx))
	title!("y2")
	plot!(xx,map(x->mu2(model,x)[1],xx))

	# println(mu2(model,x10,y20)[1])
	# println(mu2hat(x10,y20))
	writecsv("data/mk1.csv",mk1.weights)
	writecsv("data/mk2.csv",mk2.weights)
	writecsv("data/Bx.csv",Bx)
	writecsv("data/By.csv",By)

	mk1,mk2,Bx,By
end

# @time mk1,mk2,Bx,By=strict_hmm(toymodel(),100,1000);



# using ProfileView
# Profile.clear()
# @profile strict_hmm(toymodel(),30,500);
# ProfileView.view()


function general_hmm(model::LinearGaussianSSM,n=10,N=50)
	
	xx1=2*randn(N)
	function ff(x1)
		x2=rand(MvNormal(model.F*[x1],model.V))[1]
		y2=rand(MvNormal(model.G*[x2],model.W))[1]
		x2,y2
	end
	xx2=zeros(N)
	yy2=zeros(N)
	for i=1:N
		xx2[i],yy2[i]=ff(xx1[i])
	end

	Hx=HG{1}
	Hy=HG{2}
		
	Bx=sort(2*randn(n))	  #Bx does not need to be sorted
	By=sort(2*randn(n))   #By still needs to be sorted for now to be used in strict_hmm
	Bxy=[(x1,ff(x1)[2]) for x1 in 2*rand(n)]

	B1=[(xx1[i],yy2[i]) for i=1:N]
	mk1=markov_nn(Hx,Hy,B1,Bx,By)                # P(Y_{t+1}|X_t) expressed in (Bx -> By) 
	B2=[((xx1[i],yy2[i]),xx2[i]) for i=1:N]
	mk2=markov_nn(RKHS2{Hx,Hy},Hx,B2,Bxy,Bx)     # P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy -> Bx)
	B3=[(xx1[i],(xx2[i],yy2[i])) for i=1:N]
	mk3=markov_nn(Hx,RKHS2{Hx,Hy},B3,Bx,Bxy)      #P(X_t+1,Y_t+1|X_t)  expressed in (Bx -> Bxy)

	# `transition1` is P(Y_{t+1}|X_t) expressed in (Bx -> By)  
	# `transition2` is P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy) -> Bx 

	mu1hat(x1)=(proj([1.0],[x1],Bx)'*mk1.weights*Bx)[1]
	mu2hat(x2)=(proj([1.0],[x2],Bx)'*mk2.weights*By)[1]
	
	# writecsv("data/gmk1.csv",mk1.weights)
	# writecsv("data/gmk2.csv",mk2.weights)
	# writecsv("data/gmk3.csv",mk3.weights)
	# writecsv("data/gBx.csv",Bx)
	# writecsv("data/gBy.csv",By)
	# Bxy_as_array=unpack(Bxy)
	# Bxy_as_array=hcat(Bxy_as_array[1],Bxy_as_array[2])
	# writecsv("data/gBxy.csv",Bxy_as_array)

	mk1,mk2,Bx,Bxy
end





@time mk1,mk2,Bx,Bxy=general_hmm(toymodel(),10,50);


# using ProfileView
# Profile.clear()
# @profile translate2RKHS(toymodel(),15);
# ProfileView.view()


function filtrs(ini,data)
	Hx=HG{1}
	Hy=HG{2}

	mk1=readcsv("data/mk1.csv",)
	mk2=readcsv("data/mk2.csv",)
	Bx=vec(readcsv("data/Bx.csv"))
	By=vec(readcsv("data/By.csv"))

	mk1=RKHSMap(Hx,Hx,line(Bx),mk1,Bx)  # P(X_{t+1}|X_t) expressed in (Bx -> Bx)  
	mk2=RKHSMap(Hx,Hy,line(Bx),mk2,By)  # P(Y_{t+1}|X_{t+1}) expressed in (Bx -> By)

	filtr(mk1,mk2,ini,data),Bx
end


function filtrg(ini,data)
	Hx=HG{1}
	Hy=HG{2}

	mk1=readcsv("data/gmk1.csv",)
	mk2=readcsv("data/gmk2.csv",)
	mk3=readcsv("data/gmk3.csv",)
	Bx=vec(readcsv("data/gBx.csv"))
	By=vec(readcsv("data/gBy.csv"))
	Bxy_as_array=readcsv("data/gBxy.csv")
	Bxy=[(Bxy_as_array[i,1],Bxy_as_array[i,2]) for i=1:size(Bxy_as_array,1)]

	mk1=RKHSMap(Hx,Hy,line(Bx),mk1,By)  # P(Y_{t+1}|X_t) expressed in (Bx -> By) 
	mk2=RKHSMap(RKHS2{Hx,Hy},Hx,line(Bxy),mk2,Bx)  # P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy -> Bx)
	mk3=RKHSMap(Hx,RKHS2{Hx,Hy},line(Bx),mk3,Bxy)  # P(X_t+1,Y_t+1|X_t)  expressed in (Bx -> Bxy)

	fs=filtersmoother(mk1,mk2,mk3,ini,data)
	fs[1],fs[2],Bx
end

function comparefilters(T=10)
	model=toymodel()	
	datx,daty=rand(model,0.0,T)
	
	initial=MvNormal(zeros(1),ones(1,1))
	fil=filter(model, daty',initial)
	smo=smooth(model, daty',initial)

	N=50
	lambda=1.0/sqrt(N)
	ini=RKHSLeftElement(HG{1},fill(1/N,1,N),vec(rand(fil.state[1],N)))

	
	true_filter=[fil.state[t].μ[1] for t=1:T]
	true_smoother=[smo.state[t].μ[1] for t=1:T]
	pl=plot(1:T,true_filter,color=:blue,label="exact")
	plot!(1:T,true_smoother,color=:blue)

	filts,Bxs=filtrs(ini,daty)
	approx_filters=[dot(filts[:,t],Bxs) for t=1:T]
	plot!(1:T,approx_filters,color=:green,label="strict filter")
	
	filtg,smoothg,Bxg=filtrg(ini,daty)
	approx_filterg=[dot(filtg[:,t],Bxg) for t=1:T]
	approx_smoothg=[dot(smoothg[:,t],Bxg) for t=1:T]
	plot!(1:T,approx_filterg,color=:orange,label="general filter")
	plot!(1:T,approx_smoothg,color=:orange)

	display(pl)
	filtg
end

# fil=comparefilters(2);

# @time fil=comparefilters(50);


# using ProfileView
# Profile.clear()
# @profile fil,filt=comparefilters(20);
# ProfileView.view()





end