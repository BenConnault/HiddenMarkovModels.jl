######################################################################################
### Working with general stochastic matrices from E1 to E2
######################################################################################

# Normalize Stochastic Matrix
nsm(q) = q./sum(q,2)


# Random Stochastic Matrix
rsm(m,n) = rand(m,n) |> nsm
rsm(m) = rsm(m,m)

# rsm(dx::Int,dy::Int)=mapslices(x->rand(Distributions.Dirichlet(x)),ones(dx,dy),2)   # would need to add an import


# draw a sparse random stochastic matrix
function rssm(dim::Int,density=.2)
	m=Array(Float64,(dim,dim))
	for i=1:dim
		m[i,:]=Base.sprand(1,dim,density)
		m[i,rand(1:dim)]+=rand()
	end
	sparse(m./sum(m,2))	
end

######################################################################################
### Working with stochastic matrices from E1 to E1 -- Markov transtion matrices
######################################################################################



stationary(q)=stationaryt(q')

function stationaryt(tq::Array{Float64,2})
	nx=size(tq,1)
	@assert sum(tq.<0) ==0 
	@assert norm(vec(sum(tq,1)) - ones(nx)) < 1e^3 

	v=tq[1:nx-1,nx]
	p=((tq[1:nx-1,1:nx-1].-v)-eye(nx-1))\-v
	vcat(p,1-sum(p))
end


function stationary(model::DiscreteHMM,nx,ny,m=1)
	nz  = nx*ny
	qzz = reshape([qxyxy(model,ix,iy,jx,jy) for ix=1:nx,iy=1:ny,jx=1:nx,jy=1:ny],nz,nz)
	piz = stationary(qzz)
	joint_z  = reshape(transpose(piz),1,ny,nx)   #joint_z[y_{1:t-1},y_t,x_t]
	qq  = reshape(qzz,nx,ny,nx,ny)
	nyy = 1
	for i = 2:m
		temp = zeros(nyy,ny,ny,nx)
		for iy=1:ny	
			for jy=1:ny	
				A_mul_B!(view(temp,:,iy,jy,:),view(joint_z,:,iy,:),view(qq,:,iy,:,jy))
			end
		end
		nyy = nyy*ny
		joint_z = reshape(temp,nyy,ny,nx)
	end
	sum(joint_z,2)
end


doc"""
    dobrushin(q)

Compute the classical Dobrushin coefficient of a Markov transition matrix.
"""
function dobrushin(q)
    n=size(q,1)
    @assert n==size(q,2)
    a=0
    for i=1:n
        for j=i+1:n
            a=max(a,norm(q[i,:]-q[j,:],1))
        end
    end
    a/2
end





######################################################################################
### Frequently useful reparametrizations with jacobians
######################################################################################



#assumes q is square
z2q(z::Array{Float64,1})=z2q(z,Int(.5*(1+sqrt(1+4*length(z)))))


function z2q(z::Array{Float64,1},dx::Int)
	#add some size checks
	dy=Int(length(z)/dx)+1
	q=zeros(dx,dy)
	mz=reshape(z,dx,dy-1)
	for ix=1:dx
		ez=[exp(slice(mz,ix,:));1]
		q[ix,:]=ez/sum(ez)
	end
	q
end




function q2z(q::Array{Float64,2})
	dx=size(q,1)
	z=zeros(dx,dx-1)
	for ix=1:dx
		z[ix,:]=log(sub(q,ix,1:dx-1)/q[ix,dx])
	end
	vec(z)
end

# jacobian d(vec(q))/d(vec(z))
function z2qjac(z::Array{Float64,1})
	dz=length(z)  #dz=dx^2-dx
	dx=round(Int,.5*(1+sqrt(1+4*dz)))
	# println(dx)
	jac=zeros(dx,dx,dx,dx-1)
	mz=reshape(z,dx,dx-1)
	ez=zeros(dx-1)
	for ix=1:dx
		ez[:]=exp(mz[ix,:])
		sez=(1+sum(ez))
		sez2=sez^2
		for jx=1:dx-1
			for jz=1:dx-1
				jac[ix,jx,ix,jz]=((jx==jz)*ez[jz]*sez-ez[jz]*ez[jx])/sez2
			end
		end
		#jx=dx:
		for jz=1:dx-1
			jac[ix,dx,ix,jz]=-ez[jz]/sez2
		end
	end
	reshape(jac,dx*dx,dz)
end
