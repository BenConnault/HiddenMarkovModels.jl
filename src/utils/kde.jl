"""
    kde(mu,nu)

Estimate of the density of mu with respect to nu.
"""
function kde(mu::Vector{Vector{Float64}},nu::Vector{Vector{Float64}};tol=1.0,kt::Kernel=Laplace())
    n1=length(mu)
    n2=length(nu)
    yy=vcat(mu,nu)
    ky=gramian(yy,kt)
    # println(ky)
    vy,ivy=kernelbasis(ky)
    kvy=ky*vy
    mty::Matrix{Float64}=ivy*spdiagm(vcat(zeros(n1),ones(n2)/n2))*kvy
    imty=cholfact(Symmetric(mty+tol/sqrt(n1+n2)*I))
    muhat=ivy*vcat(ones(n1)/n1,zeros(n2))
    vvy=imty\muhat
    wy=kvy*vvy
    zy=sum(wy[n1+1:n1+n2])/n2
    # println(zy)
    nwy=wy/zy
    # iy=sortperm(yy)
    yy,nwy
end



"""
    ckde(mux,muy,nux,nuy;tolx=1.0,toly=1.0,ktx,kty)

Compute an estimate of the conditioal densitGiven a sample `mux,muy` is
"""
function ckde(mux::Vector{Vector{Float64}},muy::Vector{Vector{Float64}},nux::Vector{Vector{Float64}},nuy::Vector{Vector{Float64}};
                tolx=1.0,toly=1.0,ktx::Kernel=Laplace(),kty::Kernel=Laplace())
    n1=length(mux)
    @assert length(muy)==n1
    n2x=length(nux)
    n2y=length(nuy)

    xx=vcat(mux,nux)
    kx=gramian(xx,ktx)
    vx,ivx=kernelbasis(kx)
    kvx=kx*vx
    mtnux::Matrix{Float64}=ivx*spdiagm(vcat(zeros(n1),ones(n2x)/n2x))*kvx
    imtnux=cholfact(Symmetric(mtnux+tolx/sqrt(n1+n2x)*I))

    yy=vcat(muy,nuy)
    ky=gramian(yy,kty)
    vy,ivy=kernelbasis(ky)
    kvy=ky*vy
    mtnuy::Matrix{Float64}=ivy*spdiagm(vcat(zeros(n1),ones(n2y)/n2y))*kvy
    imtnuy=cholfact(Symmetric(mtnuy+toly/sqrt(n1+n2y)*I))

    muhat=ivx*sparse(1:n1,1:n1,ones(n1)/n1,n1+n2x,n1+n2y)*ivy'

    wxy=kvx*transpose(kvy*(imtnuy\transpose(imtnux\muhat)))
    ### wx is the self-normalizing constant: 
    # wx estimates f_mu(x)/f_nu(x), 
    # but regardless of how good of an estimate it is, 
    # this is the normalizing constant for the rows of wxy[x,y]=f(y|x)
    # so that each row is a density (sums to one)
    ###
    wx=vec(sum(wxy[:,n1+1:n1+n2y],2)/n2y)    
    nwxy=diagm(1./wx)*wxy
    xx,yy,nwxy
end

