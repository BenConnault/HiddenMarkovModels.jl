################################################### 
# Kernel Basis Filtering 
#
################################################### 

abstract type TO
end


struct Transition <: TO
end

struct Observation <: TO
end


rand(model,x,to::Transition)=randa(model,x)
rand(model,x,to::Observation)=randb(model,x)



function markovapprox(model,xx,yy,ky,m=1000,to::TO=Transition())
    nx=length(xx)
    ny=length(yy)
    q=zeros(nx,nx)
    for ix=1:nx
        print("$ix ")
        gi=zeros(ny)
        for j=1:m
            y=rand(model,xx[ix],to)
            for jy=1:ny
                gi[jy]=gi[jy]+kk(yy[jy],y)/m
            end            
        end
        q[ix,:]=probnorm(ky\gi)
    end
    q
end




function kbfilter(model,ini::Vector{Vector{Float64}},xx,yy,data,m=300)
    kx=gramian(xx)
    init=probnorm(kx\[mean(kk(x1,x2) for x1=ini) for x2=xx])
    kbfilter(model,init,xx,yy,data,m)
end


function kbfilter(model,ini::Vector{Float64},xx,yy,data,m=300)
    kx=gramian(xx)
    ky=gramian(yy)

    print("Building transition matrix... ")
    qxx=markovapprox(model,xx,xx,kx,m,Transition())
    println()

    print("Building observation matrix... ")
    qxy=markovapprox(model,xx,yy,ky,m,Observation())
    # qxy=markovapprox(model.observation,xx,yy,ky,m)    # TO DO: rewrite dispatch flow
    println()
    
    kbfilter(ini::Vector{Float64},xx,yy,qxx,qxy,kx,ky,data)
end


function kbfilter(ini::Vector{Float64},xx,yy,qxx,qxy,kx,ky,data,tol=1.0)
    T=length(data)
    nx=length(xx)

    fil=Array{Float64}(nx,T)
    fil[:,1]=ini
    
    print("Running the filter... ")
    for t=1:T-1
        (t%10==0)?print("$t "):nothing
        
        predictive=upq(fil[:,t],qxx)

        gy=[kk(y,data[t+1]) for y=yy]
        fil[:,t+1]=kbr(qxy,ky,predictive,gy)

    end
    println()

    fil,qxx,qxy
end


################################################### 
# Kernel Basis Filtering with exact density 
#
################################################### 


function kfbfilter(model,ini::Vector{Vector{Float64}},xx,yy,data,m=300)
    kx=gramian(xx)
    init=probnorm(kx\[mean(kk(x1,x2) for x1=ini) for x2=xx])
    kfbfilter(model,init,xx,yy,data,m)
end


function kfbfilter(model,ini::Vector{Float64},xx,yy,data,m=300)
    kx=gramian(xx)
    ky=gramian(yy)

    print("Building transition matrix... ")
    qxx=markovapprox(model,xx,xx,kx,m,Transition())
    println()

    print("Building observation matrix... ")
    qxy=markovapprox(model,xx,yy,ky,m,Observation())
    # qxy=markovapprox(model.observation,xx,yy,ky,m)    # TO DO: rewrite dispatch flow
    println()
    
    kfbfilter(model, ini::Vector{Float64},xx,yy,qxx,qxy,kx,ky,data)
end


function kfbfilter(model, ini::Vector{Float64},xx,yy,qxx,qxy,kx,ky,data)
    T=length(data)
    nx=length(xx)

    fil=Array{Float64}(nx,T)
    fil[:,1]=ini

    emi=MvNormal(zeros(length(xx[1])),model.core.W)
    emiss(x,y)=pdf(emi,y-model.core.G*x)

    print("Running the kfb filter... ")
    for t=1:T-1
        (t%10==0)?print("$t "):nothing
        
        predictive=upq(fil[:,t],qxx)

        w=[emiss(x,data[t+1]) for x=xx]
        fil[:,t+1]=upf(predictive,w)

    end
    println()

    fil,qxx,qxy
end

################################################### 
# Kernel Basis Filtering with wrong bayes step
# Yet works well in practice?
#
################################################### 




function kbfilterbis(model,ini::Vector{Vector{Float64}},xx,yy,data,m=300)
    kx=gramian(xx)
    init=probnorm(kx\[mean(kk(x1,x2) for x1=ini) for x2=xx])
    kbfilterbis(model,init,xx,yy,data,m)
end


function kbfilterbis(model,ini::Vector{Float64},xx,yy,data,m=300)
    kx=gramian(xx)
    ky=gramian(yy)

    print("Building transition matrix... ")
    qxx=markovapprox(model,xx,xx,kx,m,Transition())
    println()

    print("Building observation matrix... ")
    qxy=markovapprox(model,xx,yy,ky,m,Observation())
    # qxy=markovapprox(model.observation,xx,yy,ky,m)    # TO DO: rewrite dispatch flow
    println()
    
    kbfilterbis(ini::Vector{Float64},xx,yy,qxx,qxy,kx,ky,data)
end


function kbfilterbis(ini::Vector{Float64},xx,yy,qxx,qxy,kx,ky,data)
    T=length(data)
    nx=length(xx)

    fil=Array{Float64}(nx,T)
    fil[:,1]=ini
    
    print("Running the filter... ")
    for t=1:T-1
        (t%10==0)?print("$t "):nothing
        
        predictive=upq(fil[:,t],qxx)

        gy=[kk(y,data[t+1]) for y=yy]
        fil[:,t+1]=probnorm(kbrbis(qxy,ky,predictive,gy))

    end
    println()

    fil,qxx,qxy
end





################################################### 
# KBR Filtering 
#
################################################### 


function kbrfilter(ini::Vector{Vector{Float64}},xx,xx2,yy,data)
    kx=gramian(xx)
    init=probnorm(kx\[mean(kk(x1,x2) for x1=ini) for x2=xx])
    kbrfilter(init,xx,xx2,yy,data)
end



function kbrfilter(ini::Vector{Float64},xx1,xx2,yy,data)
    T=length(data)
    nx=length(xx1)

    kx1=gramian(xx1)
    kx12=gramian(xx1,xx2)
    kx2=gramian(xx2)
    ky=gramian(yy)

    fil=Array{Float64}(nx,T)
    fil[:,1]=ini
    
    print("Running the filter... ")
    for t=1:T-1
        (t%10==0)?print("$t "):nothing
        
        predictive=probnorm(ksr2(kx1,fil[:,t]))         #p(x_t+1|y_1:t) expressed in xx2 
        
        gx=kx2*predictive
        gy=[kk(y,data[t+1]) for y=yy]
        mu2=probnorm(kbr2(kx2,ky,gx,gy))                 #p(x_t+1|y_1:t+1) expressed in xx2
        
        fil[:,t+1]=probnorm(kx1\(kx12*mu2))

    end
    println()

    fil
end

