

# API for common update functions 


# Markov rule: nu = mu Q
upq(mu,q)=vec(mu'*q)   #un-optimized

upq!(nu,mu,q)=At_mul_B!(nu,q,mu)

# Boltzman Gibbs update: nu = (mu.*f)/sum(mu.*f)
upf(mu,f)=normalize(mu.*f,1)  #un-optimized

function upf!(nu,mu,f)
    n=length(nu)
    acc=0.0
    for i=1:n
        nu[i] = mu[i]*f[i]
        acc = acc + nu[i]
    end
    for i=1:n
        nu[i] /= acc
    end
end


upr(mu,r)=normalize(vec(mu'*r),1)  # positive operator




# Diagnostic functions

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









