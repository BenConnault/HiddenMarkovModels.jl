# API for common update functions 
upq(mu,q)=vec(mu'*q)  # Markov kernel
upf(mu,f)=normalize(mu.*f,1)  # Boltzman Gibbs update
upr(mu,r)=normalize(vec(mu'*r),1)  # positive operator


abstract type FilteringTechnique
end



function markovapprox(model,xx,yy,ky,m=1000,tol=0.1)
    nx=length(xx)
    ny=length(yy)
    q=zeros(nx,ny)
    for ix=1:nx
        print("$ix ")
        gi=zeros(ny)
        for j=1:m
            y=rand(model,xx[ix])
            for jy=1:ny
                gi[jy]=gi[jy]+kk(yy[jy],y)/m
            end            
        end
        q[ix,:]=probnorm((ky+tol*I/sqrt(ny))\gi)
    end
    q
end


#### DIAGNOSTICS

function coefvar(w)
    tot=sum(w)
    n=length(w)
    sqrt(mean((n*(w/tot)-1).^2))
end

function ess(w)
    n=length(w)
    n/(1+coefvar(w)^2)
end

function relerror(q1,q2,ky)
    nx=size(q1,1)
    err=zeros(nx)
    for ix=1:nx
        dmui=q1[ix,:]-q2[ix,:]
        err[ix]=sqrt(dot(dmui,ky*dmui))
    end
    err
end






############################ Discrete Strict Hidden Markov 



function filtr(model::DiscreteHiddenMarkovModel,data,ini)
    T=length(data)
    nx=length(ini)
    fil=Array{Float64}(nx,T)
    fil[:,1]=ini
    for t=2:T
        fil[:,t]=upf(upq(fil[:,t-1],model.a),model.b[:,data[t]])
    end
    fil
end




