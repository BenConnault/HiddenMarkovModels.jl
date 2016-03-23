# project (w1,B1) in the B2 basis, with G12=B1'B2 and G1 = B1'B1
# argmin_w2 |B1 w1-B2 w2| = < B1 w1-B2 w2 , B1 w1-B2 w2 > = 
# w1'B1'B1w1 - 2 w1' G12 w1 + w2' G2 w2 
function project_withConvexjl(w1,G12,G2,lambda=0.0,mu=0.0)
	aa=At_mul_B(G12,w1)
	w2 = Variable(size(G2,1))
	#does not make sense to penalize proportional to norm(w2,1) which is equal to 1...
	ff(w2)=quadform(w2,G2)-2*dot(aa,w2)+lambda*((1-mu)*norm(w2,1)+mu*norm(w2,2))
	problem = minimize(ff(w2), sum(w2) == 1.0, w2 >= 0.0)
	solve!(problem, SCSSolver(verbose=0))
	w2.value
end

# project (w1,B1) in the B2 basis, with G12=B1'B2 and G2 = B2'B2
# argmin_w2 |B1 w1-B2 w2| = < B1 w1-B2 w2 , B1 w1-B2 w2 > = 
# w1'B1'B1w1 - 2 w1' G12 w1 + w2' G2 w2 
function project(w1,G12,G2)
	aa=vec(At_mul_B(G12,w1))
	n=size(G2,1)
	m = Model(solver=IpoptSolver(print_level=0))
	@defVar(m, 0.0 <= w2[1:n] <= 1.0 )
	@setObjective(m, Min,  (w2'*G2*w2)[1]-2*dot(w2,aa) )
	@addConstraint(m, sum(w2) == 1 )
	status = solve(m)
	getValue(w2)
end


# project (w1,B1) in the B2 basis by projection each element of B1\
# Scalar case
function proj_nn(w1,B1,B2)
	n2=length(B2)
	w=zeros(n2)
	for i=1:length(w1)
		if B1[i]<B2[1]
			w[1]+=w1[i]
		elseif B1[i]>B2[n2]
			w[n2]+=w1[i]
		else
			range=searchsorted(B2,B1[i])
			iy=range.start
			ix=range.stop
			if ix==iy
				w[iy]+=w1[i]
			else
				lambda=(B1[i]-B2[ix])/(B2[iy]-B2[ix])
				w[ix]+=(1.0-lambda)*w1[i]
				w[iy]+=lambda*w1[i]
			end
		end
	end
	w
end
  



# project (w1,B1) on B2 by finding k nearest neighbours of each element 
# of B1 in B2, using pre-computed search tree representation of B2 `tree2`.
# G2=B2'B2 is pre-computed
function proj_nn{H<:RKHS}(::Type{H},w1,B1,B2,tree2,G2,k::Int)
	@assert size(B1,1)==length(w1)
	n2=length(B2)
	w=zeros(n2)
	aa=zeros(k)
	for i=1:length(w1)
		inns, dists = knn(tree2, [B1[i][1];B1[i][2]], k)
		Gi=G2[inns,inns]
		# @code_warntype(kernel(H,slice(B2,inns),fill(B1[i],1,1)))
		# error()
		aa=vec(kernel(H,slice(B2,inns),fill(B1[i],1,1)))
		m = Model(solver=IpoptSolver(print_level=0))
		@defVar(m, 0.0 <= w2[1:k] <= 1.0 )
		# @defVar(m, 0.0 <= w2[1:k]  )
		@setObjective(m, Min,  (w2'*Gi*w2)[1]-2*dot(w2,aa) )
		@addConstraint(m, sum(w2) == 1 )
		status = solve(m)
		w[inns]+=w1[i]*getValue(w2)
	end
	w
end

# specialized verstion of proj_nn
# project ((wx,Bx) x (1.0,delta_y)) on Bxy by finding k nearest neighbours of each element 
# of B1 in Bxy, using pre-computed search tree representation of B2 `tree2`.
# Gxy=Bxy'Bxy is also pre-computed
# Gx=(the x part of Bxy)'Bx is also pre-computed
# IN FIRST TEST NO SPEED-UP OBSERVED because most time spent in solve(.) -> get rid of this is no speed-up
function proj_nn{Hy<:RKHS}(wx,Bx,delta_y,By,::Type{Hy},Bxy,xytree,Gxy,Gx,k::Int)
	n2=length(Bxy)
	w=zeros(n2)
	aa=zeros(k)
	for i=1:length(Bx)
		inns, dists = knn(xytree, [Bx[i];delta_y], k)
		Gi=Gxy[inns,inns]
		aa=slice(Gx,inns,i).* vec(kernel(Hy,unpack(slice(Bxy,inns))[2],fill(delta_y,1,1)))
		m = Model(solver=IpoptSolver(print_level=0))
		@defVar(m, 0.0 <= w2[1:k] <= 1.0 )
		# @defVar(m, 0.0 <= w2[1:k]  )
		@setObjective(m, Min,  (w2'*Gi*w2)[1]-2*dot(w2,aa) )
		@addConstraint(m, sum(w2) == 1 )
		status = solve(m)
		w[inns]+=wx[i]*getValue(w2)
	end
	w
end

# Because data y_t, y_{t+1} will enter transition kernels as singletons, 
# there is a case to be made for biting the bullet and computing
#  kernels in kronecker basis. 
# This will be more computationally intensive but will simplify / speed up / be more accurate 

function repack(aa)
	n=length(aa)
	d1=length(aa[1][1])
	d2=length(aa[1][2])
	bb=zeros(d1+d2,n)
	for i=1:length(aa)
		bb[1:d1,i]=aa[i][1]
		bb[d1+1:d1+d2,i]=aa[i][2]
	end
	bb
end

type XYMetric{Hx,Hy} <: Metric 
	dims::Tuple{Int,Int}
end

function kernel{H <: DiscreteRKHS}(::Type{H},x::Vector{Float64},y::Vector{Float64})
	@assert length(x)==1
	@assert length(y)==1
	kernel(H,Int(x[1]),Int(y[1]))
end

function evaluate{Hx,Hy}(d::XYMetric{Hx,Hy},a::AbstractArray, b::AbstractArray)
	dx,dy=d.dims
	xya=(a[1:dx],a[dx+1:dx+dy])
	xyb=(b[1:dx],b[dx+1:dx+dy])
	# xya=(slice(a,1:dx),slice(a,dx+1:dx+dy))
	# xyb=(slice(b,1:dx),slice(b,dx+1:dx+dy))
	sqrt(kernel(RKHS2{Hx,Hy},xya,xya)+kernel(RKHS2{Hx,Hy},xyb,xyb)-2*kernel(RKHS2{Hx,Hy},xya,xyb))
end

evaluate(d::XYMetric,a::AbstractMatrix, b::AbstractArray,col::Int, do_end::Bool=true)=
	evaluate(d, slice(a, :, col), b)



function searchtree(Hx,Hy,Bxy)
	dx,dy=length(Bxy[1][1]),length(Bxy[1][2])
	bxy=repack(Bxy)
	# BruteTree(bxy, XYMetric{Hx,Hy}((dx,dy)); reorder = false)
	BallTree(bxy, XYMetric{Hx,Hy}((dx,dy)); reorder = false)
end

function filtr{Hx<:RKHS,Hy<:RKHS}(transition1::RKHSMap{Hx,Hy},transition2::RKHSMap{RKHS2{Hx,Hy},Hx},initial,data)
	# `transition1` is P(Y_{t+1}|X_t) expressed in (Bx -> By)  
	# `transition2` is P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy) -> Bx 
	# `initial` is P(X_1 | Y_1=y_1) in an arbitrary basis
	
	Bxy=vec(transition2.leftpoints)

	@assert vec(transition1.leftpoints)==transition2.rightpoints

	xytree = searchtree(Hx,Hy,Bxy)

	filter=_filtr(transition1,transition2,xytree,initial,data)

	filter
end


function filtersmoother{Hx<:RKHS,Hy<:RKHS}(transition1::RKHSMap{Hx,Hy},transition2::RKHSMap{RKHS2{Hx,Hy},Hx},
	transition::RKHSMap{Hx,RKHS2{Hx,Hy}},initial,data)
	# `transition1` is P(Y_{t+1}|X_t) expressed in (Bx -> By)  
	# `transition2` is P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy) -> Bx 
	# `transition` is P(X_t+1,Y_t+1|X_t) expressed in (Bx -> Bxy)
	# `initial` is P(X_1 | Y_1=y_1) in an arbitrary basis
	
	Bxy=transition.rightpoints

	@assert vec(transition1.leftpoints)==transition2.rightpoints
	@assert transition1.leftpoints==transition.leftpoints
	@assert vec(transition2.leftpoints)==Bxy

	xytree = searchtree(Hx,Hy,Bxy)

	filter=_filtr(transition1,transition2,xytree,initial,data)
	smoother=_smoother(transition,filter,xytree,data)

	filter,smoother
end

function _filtr{Hx<:RKHS,Hy<:RKHS}(transition1::RKHSMap{Hx,Hy},transition2::RKHSMap{RKHS2{Hx,Hy},Hx},xytree,initial,data)
	# `transition1` is P(Y_{t+1}|X_t) expressed in (Bx -> By)  
	# `transition2` is P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy) -> Bx 
	# `initial` is P(X_1 | Y_1=y_1) in an arbitrary basis

	Bx=vec(transition1.leftpoints)
	By=transition1.rightpoints
	Bxy=vec(transition2.leftpoints)
	dx,dy=length(Bxy[1][1]),length(Bxy[1][2])
	kxy=2*(dx+dy)  #number of neighbors to use

	T=length(data)
	filter=Array(Float64,length(Bx),T)
	filter[:,1]=proj_nn(vec(initial.weights),initial.points,Bx)

	mk1=transition1.weights
	mk2=transition2.weights

	# Gx=kernel(Hx,unpack(Bxy)[1],lBx)
	Gxy=kernel(RKHS2{Hx,Hy},Bxy,line(Bxy))

	for t=1:T-1

		### MK1
		# P(X_t,Y_t+1 | y_1:t)
		joint=scale(filter[:,t],mk1)

		#posterior X_t | Y_{t+1},y_1:t (given as transposed stochastic matrix)
		posterior_nucleus=joint./sum(joint,1)

		# (X_t | y_{1:t+1}) = (X_t | Y_{t+1},y_1:t) x (Y_t+1 = y_t+1)
		delta_yt1=proj_nn([1.0],[data[t+1]],By)
		# delta_yt1=proj_nn()     #TODO: move to actual nearest neighbor search for generality
		posterior=posterior_nucleus*delta_yt1   # X_t | Y_{t+1}=y_t+1 in Bx

		### MK2
		#mk2=P(X_{t+1}|X_t,Y_{t+1})

		# I want to project ((posterior,Bx) x (1.0,delta_{y_t+1}) on Bxy)  
		B=[(Bx[i],data[t+1])for i=1:length(Bx)]   #probably wasteful construction
		x1_y2=proj_nn(RKHS2{Hx,Hy},posterior,B,Bxy,xytree,Gxy,kxy)
		#specialized version (no speed-ud observed as of now):
		# x1_y2=proj_nn(posterior,Bx,data[t+1],By,Hy,Bxy,xytree,Gxy,Gx,kxy)  
		filter[:,t+1]=At_mul_B(mk2,x1_y2)

	end
	filter
end


function _smoother{Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{Hx,RKHS2{Hx,Hy}},filter,xytree,data)
	# `transition` is P(X_t+1,Y_t+1|X_t) expressed in (Bx -> Bxy)
	
	Bx=vec(transition.leftpoints)
	Bxy=transition.rightpoints
	dx,dy=length(Bxy[1][1]),length(Bxy[1][2])
	kxy=2*(dx+dy)  #number of neighbors to use

	T=length(data)
	smoother=Array(Float64,length(Bx),T)
	smoother[:,T]=filter[:,T]

	mk=transition.weights

	# Gx=kernel(Hx,unpack(Bxy)[1],lBx) #might be need for a more specialized version of proj_nn below
	Gxy=kernel(RKHS2{Hx,Hy},Bxy,line(Bxy))

	for t=T-1:-1:1
		# P(X_t+1,Y_t+1 | X_t, y_{1:t}) = Bayes(P(X_t,y_1:t),P(X_t+1,Y_t+1 | X_t, y_t))
		# P(X_t+1,Y_t+1 | X_t, y_{1:t}) = Bayes(P(X_t,y_1:t),P(X_t+1,Y_t+1 | X_t)) in the special case at hand
		# joint in (Bx -> Bxy)
		joint=scale(filter[:,t],mk)											

		# P(X_t | X_t+1,Y_t+1, y_{1:t}) in (Bx <- Bxy)  (transposed)
		bayes=joint./sum(joint,1) 

		xy_b=[(Bx[i],data[t+1]) for i=1:length(Bx)]   #probably wasteful construction
		xy=proj_nn(RKHS2{Hx,Hy},slice(smoother,:,t+1),xy_b,Bxy,xytree,Gxy,kxy)		
		smoother[:,t]=bayes*xy
	end
	smoother
end


function estep{H,Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{H,RKHS2{Hx,Hy}},initial,data;lambda=0.0)
	@assert H <: Union{Hx,RKHS2{Hx,Hy}}
	filter=filtr(transition,initial,data,lambda=lambda)

	T=length(data)

	smoother=filter[T]
	prior=filter[T-1]											
	conditional=sumrule(Dirac(Hy,data[T-1]),transition)		
	posterior=bayesrule(prior,conditional,lambda=lambda)	
	revdict=sumrule(Dirac(Hy,data[T-1+1]),posterior)
	eweights=Dict(T-1 => transpose(chainrule(smoother,revdict))) 			
	smoother=sumrule(smoother,revdict)              

	for t=T-2:-1:1

		# P(X_t | Y_{1:t}=y_{1:t})
		prior=filter[t]											

		# P(X_{t+1},Y_{t+1} | X_t, Y_t=y_t) = P(X_{t+1},Y_{t+1} | X_t, Y_{1:t}=y_{1:t})
		conditional=sumrule(Dirac(Hy,data[t]),transition)		

		# P(X_t | X_{t+1},Y_{t+1}, Y_{1:t}=y_{1:t})
		posterior=bayesrule(prior,conditional,lambda=lambda)	

		# P(X_t | X_{t+1}, Y_{1:t+1}=y_{1:t+1}) = P(X_t | X_{t+1}, Y_{1:T}=y_{1:T})
		revdict=sumrule(Dirac(Hy,data[t+1]),posterior) 			

		# P(X_t,X_{t+1} | Y_{1:T}=y_{1:T})			
		eweights[t]=transpose(chainrule(smoother,revdict)) 		

		# P(X_t | Y_{1:T}=y_{1:T})
		smoother=sumrule(smoother,revdict)              		
	end
	eweights
end

