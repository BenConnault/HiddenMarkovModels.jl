

# line(x::AbstractArray)=reshape(x,1,length(x))

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


stationary(q)=stationaryt(q')

function stationaryt(tq::Array{Float64,2})
	n=size(tq)[1]
	v=tq[1:n-1,n]
	p=((tq[1:n-1,1:n-1].-v)-eye(n-1))\-v
	vcat(p,1-sum(p))
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


# REPARAMETRIZATION
# frequently useful when optimizing over a stochastic matrix


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
