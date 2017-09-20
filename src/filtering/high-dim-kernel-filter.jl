###################################################
################# UTILITIES
################################################### 


# Exact low rank approximation
function lra(x,r)
    u,s,v=svd(x)
    u[:,1:r],s[1:r],v[:,1:r]
end

# randomized low rank approximation
function rlra(x,r)
    n1,n2=size(x)
    omega=randn(n2,r)
    y=x*omega
    q=qr(y)[1]
    b=q'*x
    bbt=b*b'
    evals,w=eig(bbt)
    svs=sqrt.(abs.(evals))
    u=q*w[:,1:r]
    v=b'*w*diagm(1./svs)[:,1:r]
    u,svs[1:r],v
end

"""
    mas()

Return a low rank factorization `(A,B,c)` such that Qf=AB(f-c'f) + c'f.
"""
function lrmarkovapprox(model,xx,yy,kx,ky,snx,m=1000,tol=1.0)
    nx,ny=length(xx),length(yy)
    iix=sort(sample(1:nx,snx,replace=false))
    sxx=xx[iix]
    skx=kx[iix,iix]
    sq=markovapprox(model,sxx,yy,ky,m,tol)
    nu=ky\ones(ny)
    nu=nu/sum(nu)
    sq1=ones(snx)*nu'
    sq0=sq-sq1
    kx[:,iix],skx\sq0,nu
end


# `q` is given as `(A,B,c)`, qg=A*B*(g-dot(c,g)) + dot(c,g)
function lrupq(mu,q)
    A,B,c=q
    Atmu=At_mul_B(A,mu)
    BtAtmu=At_mul_B(B,Atmu)
    BtAtmu+c
end

# for testing purposes
lrupq2(mu,q)=upq(mu,lrq2q(q))

function lrq2q(q)
    A,B,c=q
    nx,ny=size(A,1),size(B,2)
    A*B*(I-ones(ny)*c')+ones(nx)*c'
end

function lrkbr2(q,lky,rky,mux,gy,tol=1.0)
    q2=lrq2q(q)
    ky=lky*rky'
    kbr(q2,ky,mux,gy,tol)
end


# function kbr(q,ky,mux,gy,tol=1.0)
#     # mu=diagm(mux)*q    # you don't want to store the full joint in memory, better to recompute it on the fly below
#     muy=upq(mux,q)
#     n=length(muy)
#     dmuy=diagm(muy)
#     # pix=diagm(mux)*q*(ky\((ky*dmuy+tol/sqrt(n)*I)\gy))    #THIS MYSTERIOUSLY SEEMS TO WORK OK  
#     pix=diagm(mux)*q*((ky*dmuy+tol/sqrt(n)*I)\gy)     
#     probnorm(pix)
# end

# return (AB+alpha I)\g
function woodbury(A,B,alpha,g)
    h=B*g
    m=(alpha*I+B*A)
    (g-A*(m\h))/alpha
end

# `q` is given as `(A,B,c)`, qg=A*B*(g-dot(c,g)) + dot(c,g)
function lrkbr(q,lky,rky,mux,gy,tol=1.0)
    muy=lrupq(mux,q)
    n=length(muy)
    dmuy=diagm(muy)
    alpha=tol/sqrt(n)
    h=woodbury(lky,rky'*dmuy,alpha,gy)
    A,B,c=q
    # pix=diagm(mux)*lrq2q(q)*h
    ch=dot(c,h)
    qb=A*B*(h-ch)+ch
    pix=mux.*qb
    # probnorm(pix)
    pix/sum(pix)
end

###################################################
################# FILTER
################################################### 



struct LowRankKernelFilter <: KernelOrBasisFilter
    xx::Vector{Vector{Float64}}
    yy::Vector{Vector{Float64}}
    kx::Matrix{Float64}
    ky::Matrix{Float64}
    #ky ~ lky*rky'
    lky::Matrix{Float64}    
    rky::Matrix{Float64}
    snx::Int
    m::Int 
    tol::Float64    #regularization parameter
end

function LRKF(xx,yy,snx,tol::Float64=1.0,m::Int=500)
    kx=gramian(xx)
    ky=gramian(yy)
    u,s,v=lra(full(ky),snx)
    lky=u*diagm(s)
    rky=v
    LowRankKernelFilter(xx,yy,kx,ky,lky,rky,snx,m,tol)
end

# function LRKF(xx,yy,kx,ky,snx,tol=1.0,m=500)
#     kx=gramian(xx)
#     ky=gramian(yy)
#     u,s,v=lra(full(ky),snx)
#     lky=u*diagm(s)
#     rky=v
#     LowRankKernelFilter(xx,yy,kx,ky,lky,rky,snx,m,tol)
# end


function filtr(model,data,ini::Vector{Float64},kf::LowRankKernelFilter)
    tol=0.0

    print("Building transition matrix... ")
    qxx=lrmarkovapprox(model.transition,  kf.xx, kf.xx, kf.kx, kf.kx, kf.snx, kf.m, tol)
    println()

    print("Building observation matrix... ")
    qxy=lrmarkovapprox(model.measurement, kf.xx, kf.yy, kf.kx, kf.ky, kf.snx, kf.m, tol)
    println()
    
    filtr(model,data,ini,qxx,qxy,kf)
end



function filtr(model,data,ini::Vector{Float64},qxx,qxy,kf::LowRankKernelFilter)
    T=length(data)
    nx,ny=size(qxy[1],1),size(qxy[2],2)

    fil=Array{Float64}(nx,T)
    fil[:,1]=ini
    predictive=zeros(nx)
    gy=zeros(ny)

    print("Running the filter... ")
    for t=1:T-1
        print((t%10==0)?"$t ":"")
        
        predictive[:]=lrupq(fil[:,t],qxx)
        # placebo=lrupq2(fil[:,t],qxx)
        # println(norm(placebo-predictive)," upq")
        for iy=1:ny
            gy[iy]=kk(kf.yy[iy],data[t+1])
        end
        fil[:,t+1]=lrkbr(qxy,kf.lky,kf.rky,predictive,gy,kf.tol)
        # placebo=lrkbr2(qxy,kf.lky,kf.rky,predictive,gy,kf.tol)
        # println(norm(placebo-fil[:,t+1]))
    end
    println()

    fil,qxx,qxy
end


