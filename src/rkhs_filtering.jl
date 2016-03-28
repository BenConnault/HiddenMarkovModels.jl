# project v=(w1,B1) on B2 by finding k nearest neighbours of each element 
# of B1 in B2, using pre-computed search tree representation of B2 `tree`.
# G2=B2'B2 is pre-computed\
# argmin_w2 |B1 w1-B2 w2| = < B1 w1-B2 w2 , B1 w1-B2 w2 > = 
# w1'B1'B1w1 - 2 w1' G12 w2 + w2' G2 w2 
function project{H <: RKHS}(v::RKHSVector{H},basis <: Basis{H},k::Int)
	tree,G2=basis
	B2=tree.data
	n2=length(B2)
	w=zeros(n2)
	aa=zeros(k)
	for i=1:length(v)
		inns = knn(tree, v.basis[i], k)    #indices of nearest neighbors
		Gi=G2[inns,inns]
		G12=broadcast(kernel,slice(B2,inns),v.basis[i])   #watch out for type instability here
		m = Model(solver=IpoptSolver(print_level=0))
		@defVar(m, 0.0 <= w2[1:k] <= 1.0 )
		# @defVar(m, 0.0 <= w2[1:k]  )
		@setObjective(m, Min,  (w2'*Gi*w2)[1]-2*dot(G12,w2) )
		@addConstraint(m, sum(w2) == 1 )
		status = solve(m)
		w[inns]+=w1[i]*getValue(w2)
	end
	w
end



typealias Basis{H} Tuple{VPTree{Point{H}},Matrix{Float64}}


function filtr{Hx<:RKHS,Hy<:RKHS}(transition1::RKHSMap{Hx,Hy},transition2::RKHSMap{Tuple{Hx,Hy},Hx},initial,data)
	# `transition1` is P(Y_{t+1}|X_t) expressed in (Bx -> By)  
	# `transition2` is P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy) -> Bx 
	# `initial` is P(X_1 | Y_1=y_1) in an arbitrary basis
	
	Bx  = transition1.leftbasis
	Bxy = transition2.leftbasis

	@assert transition1.leftbasis==transition2.rightbasis

	xtree = VPTree(Bx, KernelDistance())
	Gx  = broadcast(kernel,Bx,line(Bx))
	xytree = VPTree(Bxy, KernelDistance())
	Gxy = broadcast(kernel,Bxy,line(Bxy))


	filter=_filtr(transition1,transition2,(xtree,Gx),(xytree,Gxy),initial,data)

	filter
end


function _filtr{Hx<:RKHS,Hy<:RKHS}(transition1::RKHSMap{Hx,Hy},transition2::RKHSMap{RKHS2{Hx,Hy},Hx},
	xbasis::Basis{Hx},xybasis::Basis{Tuple{Hx,Hy}},initial,data)
	# `transition1` is P(Y_{t+1}|X_t) expressed in (Bx -> By)  
	# `transition2` is P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy) -> Bx 
	# `initial` is P(X_1 | Y_1=y_1) in an arbitrary basis

	Bx    = transition1.leftbasis
	By    = transition1.rightbasis
	Bxy   = transition2.leftbasis
	# Hx,Hy = Bx[1].space,By[1].space
	dx,dy = dimension(Hx),dimension(Hy)
	kx    = 2*dx        #number of neighbors to use
	kxy   = 2*(dx+dy)  #number of neighbors to use

	T=length(data)
	filter=Array(Float64,length(Bx),T)
	filter[:,1]=project(initial,xbasis,k::Int)

	mk1 = transition1.weights
	mk2 = transition2.weights

	
	for t=1:T-1

		### MK1
		# P(X_t,Y_t+1 | y_1:t)
		joint=scale(filter[:,t],mk1)

		#posterior X_t | Y_{t+1},y_1:t (given as transposed stochastic matrix)
		posterior_nucleus=joint./sum(joint,1)

		# (X_t | y_{1:t+1}) = (X_t | Y_{t+1},y_1:t) x (Y_t+1 = y_t+1)
		delta_y=RKHSVector([1.0],[Point(Hy,data[t+1])])
		wdelta_y_in_By=project(delta_y,ybasis,ky)
		# delta_yt1=proj_nn()     #TODO: move to actual nearest neighbor search for generality
		posterior=posterior_nucleus*wdelta_y_in_By   # X_t | Y_{t+1}=y_t+1 in Bx

		### MK2
		#mk2=P(X_{t+1}|X_t,Y_{t+1})

		# I want to project mu=((posterior,Bx) x (1.0,delta_{y_t+1}) on Bxy  
		y=Point(Hy,data[t+1])
		mu=RKHSVector(posterior,[(x,y) for x in Bx])
		wx1_y2=project(mu,xybasis,kxy)
		filter[:,t+1]=At_mul_B(mk2,wx1_y2)

	end
	filter
end

function filtersmoother{Hx<:RKHS,Hy<:RKHS}(transition1::RKHSMap{Hx,Hy},transition2::RKHSMap{RKHS2{Hx,Hy},Hx},
	transition::RKHSMap{Hx,RKHS2{Hx,Hy}},initial,data)
	# `transition1` is P(Y_{t+1}|X_t) expressed in (Bx -> By)  
	# `transition2` is P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy) -> Bx 
	# `transition` is P(X_t+1,Y_t+1|X_t) expressed in (Bx -> Bxy)
	# `initial` is P(X_1 | Y_1=y_1) in an arbitrary basis
	
	Bxy=transition.rightbasis

	@assert transition1.leftbasis == transition2.rightbasis
	@assert transition1.leftbasis == transition.leftbasis
	@assert transition2.leftbasis == Bxy

	xytree = VPTree(bxy, KernelDistance())

	filter=_filtr(transition1,transition2,xytree,initial,data)
	smoother=_smoother(transition,filter,xytree,data)

	filter,smoother
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