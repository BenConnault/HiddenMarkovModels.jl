



# API for common update functions 
upq(mu,q)=vec(mu'*q)  # Markov kernel
upf(mu,f)=normalize(mu.*f,1)  # Boltzman Gibbs update
upr(mu,r)=normalize(vec(mu'*r),1)  # positive operator



############################ Strict Hidden Markov 

initial(ini,bx,cgx)=cgx\[mean(ip(Dirac(x1),x2) for x1=ini) for x2=bx]



function filtr(ini::Vector{Vector{Float64}},bx,nuy,model::StrictHiddenMarkovModel,data,ny=300)
    gx=gramian(bx)
    #wasteful because cgx will be computed twice, but this can be solved by rewriting the dispatch pipeline
    #not a priority until computational bottleneck
    cgx=cholfact(gx)     
    filtr(initial(ini,bx,cgx),bx,nuy,model,data,ny)
end

function filtr(ini::Vector{Float64},bx,nuy,model::StrictHiddenMarkovModel,data,ny=300)
    T=length(data)
    nx=length(bx)
    gx=gramian(bx)
    cgx=cholfact(gx)

    tt=time()
    print("Building Q... ")
    q=mutation(model,bx,cgx,ny)
    println()

    print("Building S... ")
    yy,s=selection(model,bx,cgx,nuy,ny)
    println()

    fil=Array{Float64}(nx,T)
    fil[:,1]=ini
    
    b=zeros(nx,nx)
    g=zeros(ny+length(nuy))
    print("Running the filter... ")
    for t=2:T
        (t%10==0)?print("$t "):nothing
        # Mutation 
        predictive=upq(fil[:,t-1],q)            
        
        # Selection
        for iy=1:length(nuy)
            g[ny+iy]=kk(data[t],nuy[iy])
        end
        for ix=1:nx
            for iy=1:ny
                g[iy]=kk(data[t],yy[ix,iy])
            end
            b[ix,:]=view(s,ix,:,:)*g   # TRUE BOTTLENECK: unrolling by hand does not speed-up
        end
        fil[:,t]=upr(predictive,b)
    
    end
    println()

    fil
end

# CholDec=LinAlg.Cholesky{Float64,Array{Float64,2}}

# can do a regularized version
function mutation(model,bx,cgx,m=1000)
    nx=length(bx)
    dx=dim(bx[1])
    gi=zeros(nx)
    nu=Sample([zeros(dx) for _=1:m])
    q=zeros(nx,nx)
    for ix=1:nx
        print("$ix ")
        for j=1:m
            nu.points[j][:]=randa(model,rand(bx[ix]))
        end
        for jx=1:nx
            gi[jx]=ip(nu,bx[jx])
        end
        q[ix,:]=normalize(cgx\gi,1)
    end
    q
end



"""
    selection(model,bx,cgx,nuy,ny)

Return the arrays of points and weights `(yy,s)` that can be used online to compute the selection matrix `b` 
once the data `yt` is observed.
Specifically, `b[i,:]` is `s[i,:,:] * k(yt,yy[i,:])`.
`nuy` is a sample from the dominating measure.
`ny` is the size of the `yy[ix,:]`, the samples that are simulated from the basis elements `bx[ix]`. 
"""
function selection(model,bx::Vector{Dirac},cgx,nuy,ny)
    nx=length(bx)
    s=zeros(nx,nx,ny+length(nuy))
    yy=Array{Vector{Float64}}(nx,ny)
    mu=Array{Vector{Float64}}(ny)
    for ix=1:nx
        print("$ix ")
        for iy=1:ny
            mu[iy]=randb(model,bx[ix].point)
        end
        yyix,s[ix,ix,:]=kde(mu,nuy)
        yy[ix,:]=yyix[1:ny]
    end
    yy,s
end

function crossinnerproducts(bx::Vector{Gauss},xx::Vector{Vector{Float64}})
    n1,n2=length(bx),length(xx)
    g=zeros(n1,n2)
    for ix=1:n1
        for jx=1:n2
            g[ix,jx]=lip(xx[jx],bx[ix].mean,bx[ix].sd)
        end
    end
    g
end

function selection(model,bx::Vector{Gauss},cgx,nuy,ny)
    nx=length(bx)
    m=length(nuy)
    s=zeros(nx,nx,ny+m)
    yy=Array{Vector{Float64}}(nx,ny)
    nux=Array{Vector{Float64}}(m+1)
    mux=Array{Vector{Float64}}(ny)
    muy=Array{Vector{Float64}}(ny)
    # fix::Matrix{Float64}
    for ix=1:nx
        print("$ix ")
        for iy=1:ny
            mux[iy]=rand(bx[ix])
            muy[iy]=randb(model,mux[iy])
        end
        for j=1:m+1
            nux[j]=bx[ix].mean+1.8*bx[ix].sd .* randn(length(bx[ix].mean))
        end
        xxix,yyix,fix= ckde(mux,muy,nux,nuy)
        cx=crossinnerproducts(bx,xxix)
        s[ix,:,:]=cgx\(cx*fix)    
        yy[ix,:]=yyix[1:ny]
    end
    yy,s
end

## Eventually we can target a more asbtract siganture like so:
# function selection{T <: NonDirac}(model,bx::Vector{T},nuy,ny)
## but a mechanism is needed to draw from a godd dominating measure for each bx[ix]




############################ Discrete Strict Hidden Markov 



function filtr(ini,model::DiscreteHiddenMarkovModel,data)
    T=length(data)
    nx=length(ini)
    fil=Array{Float64}(nx,T)
    fil[:,1]=ini
    for t=2:T
        fil[:,t]=upf(upq(fil[:,t-1],model.a),model.b[:,data[t]])
    end
    fil
end




