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



# P(Y_t+1|X_t=x1)  (called mk1 in rkhs filtering)
mu1(model::LinearGaussianSSM,x1::Vector)=model.G*model.F*x1
sigma1(model::LinearGaussianSSM,x1::Vector)=model.G*model.V*model.G'+model.W
mu1(model::LinearGaussianSSM,x1::Float64)    = mu1(model,[x1])
sigma1(model::LinearGaussianSSM,x1::Float64) = sigma1(model,[x1])
# P(X_t+1|X_t=x1,Y_{t+1}=y_2)  (called mk2 in rkhs filtering)
mu2(model::LinearGaussianSSM,x1::Vector,y2::Vector)=model.F*x1-model.V*model.G'*((model.G*model.V*model.G'+model.W)\(y2-model.G*model.F*x1))
sigma2(model::LinearGaussianSSM,x1::Vector,y2::Vector)=model.V-model.V*model.G'*((model.G*model.V*model.G'+model.W)\model.G*model.V)
mu2(model::LinearGaussianSSM,x1::Float64,y2::Float64)    = mu2(model,[x1],[y2])
sigma2(model::LinearGaussianSSM,x1::Float64,y2::Float64) = sigma2(model,[x1],[y2])
mu2(model::LinearGaussianSSM,x1y2)=mu2(model,x1y2[1],x1y2[2])
sigma2(model::LinearGaussianSSM,x1y2)=sigma2(model,x1y2[1],x1y2[2])


#N observations, n bins, alpha tuning parameter
function laplace_smooth!(w::Vector{Float64},N::Int,n::Int,alpha::Float64)
	broadcast!(+,w,w,alpha/N)
	scale!(w,1/(1+alpha*n/N))
end

function laplace_smooth(w::Vector{Float64},N::Int,n::Int,alpha::Float64)
	w2=similar(w)
	laplace_smooth!(w2,N,n,alpha)
	w2
end

#computes a markov kernel by doing a global projection of 
# the joint (1/n,B1), where B1 is a list of pairs (x_i,y_i)
# to a joint on Bxy=(Bx x By),
# then conditioning
function markov(joint_original::RKHSVector,Bx::RKHSBasis,By::RKHSBasis)
	alpha=.5
	Hx,Hy,Hxy=map(rkhs,(Bx,By,joint_original))
	@assert Hxy == (Hx,Hy)
	kxy=2*dimension(Hxy)
	Bxy=RKHSBasis(Hxy,vec([(x,y) for x=Bx.points,y=By.points]))
	xybasistree=RKHSBasisTree(Bxy)
	joint=project(joint_original,xybasistree,kxy)
	N,n=map(length,(joint_original.basis,Bxy))
	laplace_smooth!(joint,N,n,.5)
	joint=reshape(joint,length(Bx),length(By))
	mk=joint./sum(joint,2)
	RKHSMap(Bx,mk,By)
end

function markov2(joint_original::RKHSVector,Bx::RKHSBasis,By::RKHSBasis)
	alpha=.5
	Hx,Hy,Hxy=map(rkhs,(Bx,By,joint_original))
	@assert Hxy == (Hx,Hy)
	kxy=2*dimension(Hxy)
	Bxy=RKHSBasis(Hxy,vec([(x,y) for x=Bx.points,y=By.points]))
	xybasistree=RKHSBasisTree(Bxy)
	Gxy=gramian(Bxy,Bxy)
	joint=project(joint_original,Bxy,Gxy)
	joint=reshape(joint,length(Bx),length(By))
	mk=joint./sum(joint,2)
	RKHSMap(Bx,mk,By)
end


function diagnosticplot(model,mk12)
	mk1,mk2=mk12
	Bx=mk1.leftbasis
	Hx=rkhs(Bx)
	xbasistree=RKHSBasisTree(Bx)
	function mu1hat(x)
		delta_x=RKHSVector([1.0],RKHSBasis(Hx,[x]))
		(project(delta_x,xbasistree)'*mk1.weights*mk1.rightbasis.points)[1]
	end
	Bxy=mk2.leftbasis
	Hxy=rkhs(Bxy)
	xybasistree=RKHSBasisTree(Bxy)
	function mu2hat(x,y)
		delta_xy=RKHSVector([1.0],RKHSBasis(Hxy,[(x,y)]))
		(project(delta_xy,xybasistree)'*mk2.weights*mk2.rightbasis.points)[1]
	end
	xx=linspace(-5,5,30)
	plot(xx,map(mu1hat,xx))
	plot!(xx,map(x->mu1(model,x)[1],xx))

	for y2=0.0
		plot(xx,map(x->mu2hat(x,y2),xx))
		title!("y2=$y2")
		plot!(xx,map(x->mu2(model,x,y2)[1],xx))
	end
end

# n: length of the bases Bx and By
# N: simulation draws to approximate the distributions
function general_hmm(model::LinearGaussianSSM,n=10,N=50)
	
	function ff(x1)
		x2=rand(MvNormal(model.F*[x1],model.V))[1]
		y2=rand(MvNormal(model.G*[x2],model.W))[1]
		x2,y2
	end

	Hx  = GaussianRKHS{1,1.0,1}()
	Hy  = GaussianRKHS{1,1.0,2}()
	Hxy = (Hx,Hy)
	Bx  = RKHSBasis( Hx,  2*randn(n))
	By  = RKHSBasis( Hy,  2*randn(n))
	Bxy = RKHSBasis( Hxy, [(x1,ff(x1)[2]) for x1 in 2*rand(n)])

	xx1 = 2*randn(N)
	yy2 = zeros(N)
	xx2 = zeros(N)
	for i=1:N
		xx2[i],yy2[i]=ff(xx1[i])
	end

	x1_y2_joint   = RKHSVector(fill(1/N,N),RKHSBasis(Hxy,[(xx1[i],yy2[i]) for i=1:N]))
	x1y2_x2_joint = RKHSVector(fill(1/N,N),RKHSBasis((Hxy,Hx),[((xx1[i],yy2[i]),xx2[i]) for i=1:N]))
	mk1=markov2(x1_y2_joint,Bx,By)                # P(Y_{t+1}|X_t) expressed in (Bx -> By) 
	mk2=markov2(x1y2_x2_joint,Bxy,Bx)     # P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy -> Bx)

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


	mk1,mk2
end


function exact_hmm(model::LinearGaussianSSM,n=10,N=50)
	
	function ff(x1)
		x2=rand(MvNormal(model.F*[x1],model.V))[1]
		y2=rand(MvNormal(model.G*[x2],model.W))[1]
		x2,y2
	end

	Hx  = GaussianRKHS{1,1.0,1}()
	Hy  = GaussianRKHS{1,1.0,2}()
	Hxy = (Hx,Hy)
	Bx  = RKHSBasis( Hx,  2*randn(n))
	By  = RKHSBasis( Hy,  [ff(x1)[2] for x1 in Bx.points])
	Bxy = RKHSBasis( Hxy, [(x1,ff(x1)[2]) for x1 in Bx.points])
	Gx  = gramian(Bx,Bx)
	Gy  = gramian(By,By)

	println("mk1...")
	mk1=zeros(n,n)
	for i=1:n
		conditional=MvNormal(mu1(model,Bx.points[i]),sigma1(model,Bx.points[i]))
		v=RKHSVector(fill(1/N,N),RKHSBasis(Hy,[rand(conditional)[1] for l=1:N]))
		mk1[i,:]=project(v,By,Gy)
	end
	mk1=RKHSMap(Bx,mk1,By)                # P(Y_{t+1}|X_t) expressed in (Bx -> By) 
	
	println("mk2...")
	mk2=zeros(n,n)
	for i=1:n
		conditional=MvNormal(mu2(model,Bxy.points[i]),sigma2(model,Bxy.points[i]))
		v=RKHSVector(fill(1/N,N),RKHSBasis(Hx,[rand(conditional)[1] for l=1:N]))
		mk2[i,:]=project(v,Bx,Gx)
	end
	mk2=RKHSMap(Bxy,mk2,Bx)     # P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy -> Bx)
	
	mk1,mk2
end

model=toymodel()

# diagnosticplot(model,general_hmm(model,30,1000))
# diagnosticplot(model,exact_hmm(model,50,1000))

# @time mk1,mk2=general_hmm(toymodel(),30,1000);


# using ProfileView
# Profile.clear()
# @profile mk1,mk2=general_hmm(toymodel(),10,50);
# ProfileView.view()



# function filtr(ini,data)
	
# 	mk1,mk2=general_hmm(toymodel(),)
# 	mk1=RKHSMap(Hx,Hy,line(Bx),mk1,By)  # P(Y_{t+1}|X_t) expressed in (Bx -> By) 
# 	mk2=RKHSMap(RKHS2{Hx,Hy},Hx,line(Bxy),mk2,Bx)  # P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy -> Bx)
# 	mk3=RKHSMap(Hx,RKHS2{Hx,Hy},line(Bx),mk3,Bxy)  # P(X_t+1,Y_t+1|X_t)  expressed in (Bx -> Bxy)

# 	fs=filtersmoother(mk1,mk2,mk3,ini,data)
# 	fs[1],fs[2],Bx
# end

function comparefilters(T=10,n=10,N=50)
	model=toymodel()	
	datx,daty=rand(model,0.0,T)
	
	initial=MvNormal(zeros(1),ones(1,1))
	fil=filter(model, daty',initial)
	smo=smooth(model, daty',initial)

	Hx=GaussianRKHS{1,1.0,1}()
	N=100
	ini=RKHSVector(fill(1/N,N),RKHSBasis(Hx,vec(rand(fil.state[1],N))))

	
	true_filter=[fil.state[t].μ[1] for t=1:T]
	true_smoother=[smo.state[t].μ[1] for t=1:T]
	pl=plot(1:T,true_filter,color=:blue,label="exact")
	# plot!(1:T,true_smoother,color=:blue)

	# filts,Bxs=filtrs(ini,daty)
	# approx_filters=[dot(filts[:,t],Bxs) for t=1:T]
	# plot!(1:T,approx_filters,color=:green,label="strict filter")
	
	mk1,mk2=exact_hmm(model,n,N)
	filt=filtr(mk1,mk2,ini,daty)

	Bx=mk1.leftbasis
	approx_filter=[dot(filt[:,t],Bx.points) for t=1:T]
	# approx_smoothg=[dot(smoothg[:,t],Bxg) for t=1:T]
	plot!(1:T,approx_filter,color=:orange,label="general filter")
	display(pl)

	# plot!(1:T,approx_smoothg,color=:orange)

	mk1,mk2
end

mk1,mk2=comparefilters(2,5,10);

# @time fil=comparefilters(50,10,100);
# @time fil=comparefilters(50,20,100);
# @time fil=comparefilters(50,10,1000);
# @time fil=comparefilters(50,20,1000);


using ProfileView
Profile.clear()
@profile comparefilters(100,30,1000);
ProfileView.view()





end