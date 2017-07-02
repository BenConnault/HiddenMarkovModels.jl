struct BasisFilter <: KernelOrBasisFilter
    xx::Vector{Vector{Float64}}
    yy::Vector{Vector{Float64}}
    kx::Matrix{Float64}
    ky::Matrix{Float64}
    m::Int
end


function BF(xx,yy,m=500)
    kx=gramian(xx)
    ky=gramian(yy)
    BasisFilter(xx,yy,kx,ky,m)
end


function filtr(model,data,ini::Vector{Float64},bf::BasisFilter)

    print("Building transition matrix... ")
    qxx=markovapprox(model.transition,bf.xx,bf.xx,bf.kx,bf.m)
    println()
    
    filtr(model,data,ini,qxx,bf)
end


function filtr(model,data,ini::Vector{Float64},qxx,bf::BasisFilter)
    T=length(data)
    nx=size(qxx,1)

    fil=Array{Float64}(nx,T)
    fil[:,1]=ini
    w=zeros(nx)

    print("Running the kfb filter... ")
    for t=1:T-1
        (t%10==0)?print("$t "):nothing
        
        predictive=upq(fil[:,t],qxx)

        for ix=1:nx
            w[ix]=cpdf(model.measurement,bf.xx[ix],data[t+1])
        end
        fil[:,t+1]=upf(predictive,w)

    end
    println()

    fil,qxx
end
