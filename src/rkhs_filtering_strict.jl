
#strict HMM
function filtr{Hx<:RKHS,Hy<:RKHS}(transition::RKHSMap{Hx,Hx},emission::RKHSMap{Hx,Hy},initial,data)
	# `transition` is P(X_{t+1}|X_t) expressed in (Bx -> Bx)  
	#   `emission` is P(Y_{t+1}|X_{t+1}) expressed in (Bx -> By)
	# `initial` is P(X_1 | Y_1=y_1)

	Bx=vec(transition.leftpoints)
	By=emission.rightpoints
	@assert Bx==transition.rightpoints
	@assert Bx==vec(emission.leftpoints)
	
	T=length(data)
	filter=Array(Float64,length(Bx),T)
	filter[:,1]=proj_nn(vec(initial.weights),initial.points,Bx)

	mk1=transition.weights
	mk2=emission.weights
	

	for t=1:T-1
		predict=At_mul_B(mk1,filter[:,t])		#P(X_t+1 | X_t ~ filter) = P(X_t+1 | y_1:t)
		joint=scale(predict,mk2)				#P(Y_t+1 , X_t+1 | y_1:t )
		posterior_kernel=joint./sum(joint,1)    #P(X_t+1 | Y_t+1,y_1:t ) in Bx <- By
		delta_y=proj_nn([1.0],[data[t+1]],By)   
		filter[:,t+1]=posterior_kernel*delta_y 		#P(X_t+1 | y_1:t+1) in Bx
	end
	filter
end