abstract type Kernel
end


struct Laplace <: Kernel
end

# kk(x1::Int,x2::Int,place_holder=Laplace()) = 1.0*(x1==x2)
kk(x1::Int,x2::Int,place_holder=Laplace()) = exp(-abs(x1-x2))

# NOTE: for now I went with a design where I am not storing or using Symmetric kernel matrices
#       (this shows in the FilteringTechnique types that carry the kernels in memory )
#       If I change my mind I can uncomment below
# function gramian(yy,kt::Kernel=Laplace())
#     ny=length(yy)
#     ky=zeros(ny,ny)
#     for iy=1:ny
#         for jy=iy:ny
#             ky[iy,jy]=kk(yy[iy],yy[jy],kt)
#         end
#     end
#     Symmetric(ky)
# end

gramian(yy,kt::Kernel=Laplace())=gramian(yy,yy,kt)


function gramian(yy1,yy2,kt::Kernel=Laplace())
    n1=length(yy1)
    n2=length(yy2)
    ky=zeros(n1,n2)
    for iy=1:n1
        for jy=1:n2
            ky[iy,jy]=kk(yy1[iy],yy2[jy],kt)
        end
    end
    ky
end



function kk(x1::Vector{Float64},x2::Vector{Float64},kt::Laplace=Laplace())
    n=length(x1)
    acc::Float64=abs(x1[1]-x2[1])
    for i=2:n
        acc=acc+abs(x1[i]-x2[i])
    end
    # exp(-acc^1.5)
    exp(-acc)
end




probnorm(mu)=normalize(mu.*(mu.>0),1)


"""
    kbr(q,ky,mux,gy)

Compute the kernel Bayes rule `prior mux -> posterior conditional on data y0`
based on a precomputed transition function matrix from `bxx` to `byy`.
`byy` has Gramian `ky`.
`mux` the coordinate vector of the prior in `bxx`.
`gy[j]` gives the inner product of the data `k_y0` with ``k_{y_j}``, 
ie. `gy[j]=k(yy[j],y0)`. 
Return the coordinate vector of the posterior in `bxx`.
"""
function kbr(q,ky,mux,gy,tol=1.0)
    # mu=diagm(mux)*q    # you don't want to store the full joint in memory, better to recompute it on the fly below
    muy=upq(mux,q)
    n=length(muy)
    dmuy=diagm(muy)
    # pix=diagm(mux)*q*(ky\((ky*dmuy+tol/sqrt(n)*I)\gy))    #THIS MYSTERIOUSLY SEEMS TO WORK OK  
    # pix=diagm(mux)*q*((dmuy*ky+tol/sqrt(n)*I)\gy)           #THIS DOES NOT WORK, TESTED ON DISCRETE CASE    
    pix=diagm(mux)*q*((ky*dmuy+tol/sqrt(n)*I)\gy)           #THIS IS IS THE REAL ALGORITHM, TESTED     
    probnorm(pix)
end





"""
    ksr2(kx,gx)

Compute the kernel sum rule based on a fixed sample ``(x,y)_{1:n}``.
`xx` has Gramian `kx`.
`gx[i]` gives the inner product of the input prior `mu` with ``k_{x_i}``. 
If `mu` is given with coordinates `b` in `xx`, then `g:=kx*b`.
Return the coordinates of the posterior in `yy`.
"""
function ksr2(kx,gx,tol=1.0)
    n=size(kx,2)
    (kx+tol/sqrt(n)*n*I)\gx    
end

"""
    kbr2(kx,ky,gx,gy)

Compute the kernel Bayes rule based on a fixed sample ``(x,y)_{1:n}``.
`xx` and `yy` have Gramians `kx` and `ky`.
`gx[i]` gives the inner product of the input prior `mu` with ``k_{x_i}``. 
If `mu` is given with coordinates `b` in `xx`, then `g:=kx*b`.
`gy[j]` gives the inner product of the data `k_y` with ``k_{y_j}``, 
ie. `gy[j]=k(yy[j],y)`. 
Return the coordinates of the posterior in `xx`.
"""
function kbr2(kx,ky,gx,gy,tol=1.0)
    mu=ksr2(kx,gx)
    n=length(mu)
    # `mu` is the coordinates the prior times conditional
    # ie both of the marginal on `y` in the y_j basis,
    # and of the joint in the (x,y)_j basis.
    dmu=diagm(mu)
    pix=dmu*ky*(((dmu*ky)^2+tol/sqrt(n)*I)\(dmu*gy))
    pix
end




##########################################################################################
## BLUEPRINT FOR A STRONGER TYPE DESIGN
##
## Would be useful to write a generic algorithm where basis elements can be anything
##########################################################################################



# abstract type ProbabilityVector
# end

# function gramian{T <: ProbabilityVector}(bx::Vector{T},kt::Laplace)
#     nx=length(bx)
#     g=ones(nx,nx)
#     for ix=1:nx-1
#         for jx=ix+1:nx
#             g[ix,jx]=ip(bx[ix],bx[jx])
#         end
#     end
#     Symmetric(g)
# end



# function kernelbasis(kx::Symmetric{Float64,Array{Float64,2}})::Tuple{Matrix{Float64},Matrix{Float64}}
#     ivx=chol(kx)
#     vx=full(copy(ivx))
#     LinAlg.LAPACK.trtri!('U', 'N', vx)   #compute the inverse
#     full(vx),full(ivx)
# end



# # k(x1::Vector{Float64},x2::Vector{Float64})=exp(-sum(abs.(x1-x2)))



# abstract type NonDirac <: ProbabilityVector
# end

# struct Dirac <: ProbabilityVector
#     point::Vector{Float64}
# end

# rand(nu::Dirac)=nu.point

# dim(nu::Dirac)=length(nu.point)

# struct Gauss <: NonDirac
#     mean::Vector{Float64}
#     sd::Vector{Float64}
#     sample::Vector{Vector{Float64}}
# end

# Gauss(mu,sigma,m::Int=1000)=Gauss(mu,sigma,[mu+sigma .* randn(length(mu)) for _=1:m])



# rand(nu::Gauss)=nu.mean+nu.sd .* randn(length(nu.mean))

# dim(nu::Gauss)=length(nu.mean)


# struct Sample <: NonDirac
#     points::Vector{Vector{Float64}}
# end

# rand(nu::Sample)=sample(nu.points)

# dim(nu::Sample)=length(nu.points[1])





# ip(v1::Dirac,v2::Dirac)=k(v1.point,v2.point)

# function ip(v1::Sample,v2::Dirac)
#     m=length(v1.points)
#     acc=0.0
#     for i=1:m
#         acc=acc+k(v1.points[i],v2.point)/m
#     end
#     acc
# end

# ip(v1::Dirac,v2::Sample)=ip(v2,v1)

# function ip(v1::Sample,v2::Sample)
#     m1=length(v1.points)
#     m2=length(v2.points)
#     acc=0.0
#     for i1=1:m1
#         for i2=1:m2
#             acc=acc+k(v1.points[i1],v2.points[i2])/(m1*m2)
#         end
#     end
#     acc
# end

# ip(v1::Dirac,v2::Gauss)=lip(v1.point,v2.mean,v2.sd)

# ip(v1::Gauss,v2::Dirac)=ip(v2,v1)

# function ip(v1::Gauss,v2::Gauss)
#     dx=dim(v2)
#     m=length(v1.sample)
#     prod=1.0
#     for i=1:dx
#         acc=0.0
#         for j=1:m
#             acc+=lip(v1.sample[j][i],v2.mean[i],v2.sd[i])/m
#         end
#         prod=prod*acc
#     end
#     prod
# end

# function ip(v1::Sample,v2::Gauss)
#     m=length(v1.points)
#     acc=0.0
#     for i=1:m
#         acc=acc+lip(v1.points[i],v2.mean,v2.sd)/m
#     end
#     acc
# end

# ip(v1::Gauss,v2::Sample)=ip(v2,v1)


# # inner product between point and independent multivariate normal 
# # sigma is the vector of standard deviations 
# function lip(x::Vector{Float64},mu::Vector{Float64},sigma::Vector{Float64})::Float64
#     prod=1.0
#     for i=1:length(x)
#         prod=prod*lip(x[i],mu[i],sigma[i])
#     end
#     prod
# end

# # inner product between point and normal for the Laplace kernel
# function lip(x::Float64,mu::Float64,sigma::Float64)::Float64
#     c1=sigma-(mu-x)/sigma
#     c2=-sigma-(mu-x)/sigma
#     exp(0.5*sigma^2+log((1-normcdf(c1))*exp(-(mu-x))+normcdf(c2)*exp(mu-x)))
# end
