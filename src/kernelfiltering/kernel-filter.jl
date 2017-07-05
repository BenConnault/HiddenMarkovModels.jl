# Prototype KernelFilter: no low-rank approximations, high memory consumption.

abstract type KernelOrBasisFilter <: FilteringTechnique
end

struct KernelFilter <: KernelOrBasisFilter
    xx::Vector{Vector{Float64}}
    yy::Vector{Vector{Float64}}
    kx::Matrix{Float64}
    ky::Matrix{Float64}
    m::Int          # number of simulation draws used for each row
    tol::Float64    # regularization parameter
end

function KF(xx,yy,m=500,tol=1.0)
    kx=gramian(xx)
    ky=gramian(yy)
    KernelFilter(xx,yy,kx,ky,m,tol)
end



function filtr(model,data,ini::Vector{Vector{Float64}},kf::KernelOrBasisFilter)
    n=length(ini)
    g=gramian(kf.xx,ini)*ones(n)/n
    init=probnorm(kf.kx\g)
    filtr(model,data,init,kf)
end


function filtr(model,data,ini::Vector{Float64},kf::KernelFilter)

    print("Building transition matrix... ")
    qxx=markovapprox(model.transition,  kf.xx, kf.xx, kf.kx, kf.m, kf.tol)
    println()

    print("Building observation matrix... ")
    qxy=markovapprox(model.measurement, kf.xx, kf.yy, kf.ky, kf.m, kf.tol)
    println()
    
    filtr(model,data,ini,qxx,qxy,kf)
end



function filtr(model,data,ini,qxx,qxy,kf::KernelFilter)
    T=length(data)
    nx,ny=size(qxy)

    fil=Array{Float64}(nx,T)
    fil[:,1]=ini
    predictive=zeros(nx)
    gy=zeros(ny)

    print("Running the filter... ")
    for t=1:T-1
        print((t%10==0)?"$t ":"")
        
        predictive[:]=upq(fil[:,t],qxx)

        for iy=1:ny
            gy[iy]=kk(kf.yy[iy],data[t+1])
        end
        fil[:,t+1]=kbr(qxy,kf.ky,predictive,gy,kf.tol)

    end
    println()

    fil,qxx,qxy
end

