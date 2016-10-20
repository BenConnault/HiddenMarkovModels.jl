abstract FilteringAlgorithm

immutable Strict  <: FilteringAlgorithm end
immutable General <: FilteringAlgorithm end
immutable Alt     <: FilteringAlgorithm end


# Morally I would like the following signature:
# function filtr{Hx<:RKHS,Hy<:RKHS}(transition1::RKHSMap{Hx,Hy},transition2::RKHSMap{Tuple{Hx,Hy},Hx},initial,data)
# BUT with my current implementation of RKHS as nested tuples of limited depth,
# If Hx and Hy are RKHS, Tuple{Hx,Hy} cannot be a nested tuple of the same maximum depth

function filtr{Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{Hx,Hx},emission::RKHSMap{Hx,Hy},initial,data,filteringalgo::Strict)
	# `transition` is P(X_{t+1}|X_t) expressed in (Bx -> Bx)  
	# `emission` is P(Y_t|X_t) expressed in (Bx -> By) 
	# `initial` is P(X_1 | Y_1=y_1) in an arbitrary basis
	
	Bx  = emission.leftbasis
	By  = emission.rightbasis

	@assert Bx==transition.rightbasis
	@assert Bx==transition.leftbasis

	xbasistree  = RKHSBasisTree(Bx)
	ybasistree  = RKHSBasisTree(By)

	filter=_filtr(transition,emission,xbasistree,ybasistree,initial,data,filteringalgo)

	filter
end

#direct acces to core algorithm if you already have the trees
function _filtr{Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{Hx,Hx},emission::RKHSMap{Hx,Hy},
	xbasistree::RKHSBasisTree{Hx},ybasistree::RKHSBasisTree{Hy},initial,data,filteringalgo::Strict)
	# `transition` is P(X_{t+1}|X_t) expressed in (Bx -> Bx)  
	# `emission` is P(Y_t|X_t) expressed in (Bx -> By) 
	# `initial` is P(X_1 | Y_1=y_1) in an arbitrary basis

	Bx  = emission.leftbasis
	By  = emission.rightbasis

	@assert Bx==transition.rightbasis
	@assert Bx==transition.leftbasis
	@assert Bx.points == xbasistree.tree.data
	@assert By.points == ybasistree.tree.data
	
	dx,dy = dimension(rkhs(xbasistree)),dimension(rkhs(ybasistree))
	kx    = 2*dx        #number of neighbors to use
	ky    = 2*dy        #number of neighbors to use

	T=length(data)
	filter=Array(Float64,length(Bx),T)
	filter[:,1]=project(initial,xbasistree,kx::Int)

	mkt = transition.weights   #P(X_{t+1}|X_t)
	mke = emission.weights
	
	for t=1:T-1

		### MK1
		# p(X_{t+1}| y_1:t) =  p(X_t | y_1:t) [T] mkt  
		pred=At_mul_B(mkt,view(filter,:,t))

		# p(X_{t+1}, Y_{t+1} | y_1:t) = p(X_{t+1}| y_1:t) [J] mke 
		joint=Diagonal(pred)*mke

		# p(X_{t+1}  | Y_{t+1}, y_1:t) (given as transposed stochastic matrix) 
		posterior_kernel=joint./sum(joint,1)

		# p(X_{t+1}  | y_{t+1}, y_1:t) = delta_{y_t+1} [T] p(X_{t+1}  | Y_{t+1}, y_1:t)
		delta_y=RKHSVector([1.0],RKHSBasis(rkhs(Hy),[data[t+1]]))
		wdelta_y_in_By=project(delta_y,ybasistree,ky)
		filter[:,t+1]=posterior_kernel*wdelta_y_in_By  

	end
	filter
end



function filtr{Hx<:RKHS,Hy<:RKHS,Hxy<:RKHS}(transition1::RKHSMap{Hx,Hy},transition2::RKHSMap{Hxy,Hx},initial,data)
	# `transition1` is P(Y_{t+1}|X_t) expressed in (Bx -> By)  
	# `transition2` is P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy -> Bx) 
	# `initial` is P(X_1 | Y_1=y_1) in an arbitrary basis
	
	@assert Hxy == Tuple{Hx,Hy}	

	Bx  = transition1.leftbasis
	By  = transition1.rightbasis
	Bxy = transition2.leftbasis

	@assert Bx==transition2.rightbasis

	xbasistree  = RKHSBasisTree(Bx)
	ybasistree  = RKHSBasisTree(By)
	xybasistree = RKHSBasisTree(Bxy)

	filter=_filtr(transition1,transition2,xbasistree,ybasistree,xybasistree,initial,data)

	filter
end


function _filtr{Hx<:RKHS,Hy<:RKHS,Hxy<:RKHS}(transition1::RKHSMap{Hx,Hy},transition2::RKHSMap{Hxy,Hx},
	xbasistree::RKHSBasisTree{Hx},ybasistree::RKHSBasisTree{Hy},xybasistree::RKHSBasisTree{Hxy},initial,data)
	# `transition1` is P(Y_{t+1}|X_t) expressed in (Bx -> By)  
	# `transition2` is P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy) -> Bx 
	# `initial` is P(X_1 | Y_1=y_1) in an arbitrary basis

	@assert Hxy == Tuple{Hx,Hy}	
	Bx=transition1.leftbasis
	@assert Bx.points == xbasistree.tree.data
	@assert Bx == transition2.rightbasis
	@assert transition1.rightbasis.points == ybasistree.tree.data
	@assert transition2.leftbasis.points == xybasistree.tree.data
	
	dx,dy = dimension(rkhs(xbasistree)),dimension(rkhs(ybasistree))
	kx    = 2*dx        #number of neighbors to use
	ky    = 2*dy        #number of neighbors to use
	kxy   = 2*(dx+dy)   #number of neighbors to use

	T=length(data)
	filter=Array(Float64,length(Bx),T)
	filter[:,1]=project(initial,xbasistree,kx::Int)

	mk1 = transition1.weights
	mk2 = transition2.weights
	
	for t=1:T-1

		### MK1
		# P(X_t,Y_t+1 | y_1:t)
		joint=Diagonal(view(filter,:,t))*mk1

		#posterior X_t | Y_{t+1},y_1:t (given as transposed stochastic matrix)
		posterior_nucleus=joint./sum(joint,1)

		# (X_t | y_{1:t+1}) = (X_t | Y_{t+1},y_1:t) x (Y_t+1 = y_t+1)
		delta_y=RKHSVector([1.0],RKHSBasis(rkhs(Hy),[data[t+1]]))
		wdelta_y_in_By=project(delta_y,ybasistree,ky)
		posterior=posterior_nucleus*wdelta_y_in_By   # X_t | Y_{t+1}=y_t+1 in Bx

		### MK2
		#mk2=P(X_{t+1}|X_t,Y_{t+1})

		# I want to project mu=((posterior,Bx) x (1.0,delta_{y_t+1}) on Bxy  
		mu=RKHSVector(posterior,RKHSBasis(rkhs(Hxy),[(x,data[t+1]) for x in Bx.points]))
		wx1_y2=project(mu,xybasistree,kxy)
		filter[:,t+1]=At_mul_B(mk2,wx1_y2)

	end
	filter
end

# function filtersmoother{Hx<:RKHS,Hy<:RKHS}(transition1::RKHSMap{Hx,Hy},transition2::RKHSMap{RKHS2{Hx,Hy},Hx},
# 	transition::RKHSMap{Hx,RKHS2{Hx,Hy}},initial,data)
# 	# `transition1` is P(Y_{t+1}|X_t) expressed in (Bx -> By)  
# 	# `transition2` is P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy) -> Bx 
# 	# `transition` is P(X_t+1,Y_t+1|X_t) expressed in (Bx -> Bxy)
# 	# `initial` is P(X_1 | Y_1=y_1) in an arbitrary basis
	
# 	Bxy=transition.rightbasis

# 	@assert transition1.leftbasis == transition2.rightbasis
# 	@assert transition1.leftbasis == transition.leftbasis
# 	@assert transition2.leftbasis == Bxy

# 	xytree = VPTree(bxy, KernelDistance())

# 	filter=_filtr(transition1,transition2,xytree,initial,data)
# 	smoother=_smoother(transition,filter,xytree,data)

# 	filter,smoother
# end



# function _smoother{Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{Hx,RKHS2{Hx,Hy}},filter,xytree,data)
# 	# `transition` is P(X_t+1,Y_t+1|X_t) expressed in (Bx -> Bxy)
	
# 	Bx=vec(transition.leftpoints)
# 	Bxy=transition.rightpoints
# 	dx,dy=length(Bxy[1][1]),length(Bxy[1][2])
# 	kxy=2*(dx+dy)  #number of neighbors to use

# 	T=length(data)
# 	smoother=Array(Float64,length(Bx),T)
# 	smoother[:,T]=filter[:,T]

# 	mk=transition.weights

# 	# Gx=kernel(Hx,unpack(Bxy)[1],lBx) #might be need for a more specialized version of proj_nn below
# 	Gxy=kernel(RKHS2{Hx,Hy},Bxy,line(Bxy))

# 	for t=T-1:-1:1
# 		# P(X_t+1,Y_t+1 | X_t, y_{1:t}) = Bayes(P(X_t,y_1:t),P(X_t+1,Y_t+1 | X_t, y_t))
# 		# P(X_t+1,Y_t+1 | X_t, y_{1:t}) = Bayes(P(X_t,y_1:t),P(X_t+1,Y_t+1 | X_t)) in the special case at hand
# 		# joint in (Bx -> Bxy)
# 		joint=scale(filter[:,t],mk)											

# 		# P(X_t | X_t+1,Y_t+1, y_{1:t}) in (Bx <- Bxy)  (transposed)
# 		bayes=joint./sum(joint,1) 

# 		xy_b=[(Bx[i],data[t+1]) for i=1:length(Bx)]   #probably wasteful construction
# 		xy=proj_nn(RKHS2{Hx,Hy},slice(smoother,:,t+1),xy_b,Bxy,xytree,Gxy,kxy)		
# 		smoother[:,t]=bayes*xy
# 	end
# 	smoother
# end