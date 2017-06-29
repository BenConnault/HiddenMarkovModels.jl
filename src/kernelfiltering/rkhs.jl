abstract type Kernel
end


struct Laplace <: Kernel
end


function kk(x1::Vector{Float64},x2::Vector{Float64},kt::Laplace=Laplace())
    n=length(x1)
    acc=abs(x1[1]-x2[1])
    for i=2:n
        acc=acc+abs(x1[i]-x2[i])
    end
    # exp(-acc^1.5)
    exp(-acc)
end



function gramian(yy::Vector{Vector{Float64}},kt::Kernel=Laplace())
    ny=length(yy)
    println(ny)
    ky=zeros(ny,ny)
    for iy=1:ny
        for jy=iy:ny
            ky[iy,jy]=kk(yy[iy],yy[jy],kt)
        end
    end
    Symmetric(ky)
end


function gramian(yy1::Vector{Vector{Float64}},yy2::Vector{Vector{Float64}},kt::Kernel=Laplace())
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


##### TYPE DESIGN for Laplace kernel

abstract type ProbabilityVector
end

function gramian{T <: ProbabilityVector}(bx::Vector{T},kt::Laplace)
    nx=length(bx)
    g=ones(nx,nx)
    for ix=1:nx-1
        for jx=ix+1:nx
            g[ix,jx]=ip(bx[ix],bx[jx])
        end
    end
    Symmetric(g)
end



function kernelbasis(kx::Symmetric{Float64,Array{Float64,2}})::Tuple{Matrix{Float64},Matrix{Float64}}
    ivx=chol(kx)
    vx=full(copy(ivx))
    LinAlg.LAPACK.trtri!('U', 'N', vx)   #compute the inverse
    full(vx),full(ivx)
end



# k(x1::Vector{Float64},x2::Vector{Float64})=exp(-sum(abs.(x1-x2)))



abstract type NonDirac <: ProbabilityVector
end

struct Dirac <: ProbabilityVector
    point::Vector{Float64}
end

rand(nu::Dirac)=nu.point

dim(nu::Dirac)=length(nu.point)

struct Gauss <: NonDirac
    mean::Vector{Float64}
    sd::Vector{Float64}
    sample::Vector{Vector{Float64}}
end

Gauss(mu,sigma,m::Int=1000)=Gauss(mu,sigma,[mu+sigma .* randn(length(mu)) for _=1:m])



rand(nu::Gauss)=nu.mean+nu.sd .* randn(length(nu.mean))

dim(nu::Gauss)=length(nu.mean)


struct Sample <: NonDirac
    points::Vector{Vector{Float64}}
end

rand(nu::Sample)=sample(nu.points)

dim(nu::Sample)=length(nu.points[1])





ip(v1::Dirac,v2::Dirac)=k(v1.point,v2.point)

function ip(v1::Sample,v2::Dirac)
    m=length(v1.points)
    acc=0.0
    for i=1:m
        acc=acc+k(v1.points[i],v2.point)/m
    end
    acc
end

ip(v1::Dirac,v2::Sample)=ip(v2,v1)

function ip(v1::Sample,v2::Sample)
    m1=length(v1.points)
    m2=length(v2.points)
    acc=0.0
    for i1=1:m1
        for i2=1:m2
            acc=acc+k(v1.points[i1],v2.points[i2])/(m1*m2)
        end
    end
    acc
end

ip(v1::Dirac,v2::Gauss)=lip(v1.point,v2.mean,v2.sd)

ip(v1::Gauss,v2::Dirac)=ip(v2,v1)

function ip(v1::Gauss,v2::Gauss)
    dx=dim(v2)
    m=length(v1.sample)
    prod=1.0
    for i=1:dx
        acc=0.0
        for j=1:m
            acc+=lip(v1.sample[j][i],v2.mean[i],v2.sd[i])/m
        end
        prod=prod*acc
    end
    prod
end

function ip(v1::Sample,v2::Gauss)
    m=length(v1.points)
    acc=0.0
    for i=1:m
        acc=acc+lip(v1.points[i],v2.mean,v2.sd)/m
    end
    acc
end

ip(v1::Gauss,v2::Sample)=ip(v2,v1)


# inner product between point and independent multivariate normal 
# sigma is the vector of standard deviations 
function lip(x::Vector{Float64},mu::Vector{Float64},sigma::Vector{Float64})::Float64
    prod=1.0
    for i=1:length(x)
        prod=prod*lip(x[i],mu[i],sigma[i])
    end
    prod
end

function lip(x::Float64,mu::Float64,sigma::Float64)::Float64
    c1=sigma-(mu-x)/sigma
    c2=-sigma-(mu-x)/sigma
    exp(0.5*sigma^2+log((1-normcdf(c1))*exp(-(mu-x))+normcdf(c2)*exp(mu-x)))
end
