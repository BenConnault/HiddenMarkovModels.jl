######################################################################################
### Various reshaping
######################################################################################

function mat{S}(xx::Vector{Vector{S}})
    T = length(xx)
    n = length(xx[1])
    yy = zeros(S,n,T)
    for t=1:T
        yy[:,t] = xx[t]
    end
    yy 
end

function unmat{S}(xx::Matrix{S})
    n,T = size(xx)
    yy = Array{Vector{S}}(T)
    for t=1:T
        yy[t] = xx[:,t]
    end
    yy
end

function tensor{Tx,Ty}(bxx::Vector{Tx},byy::Vector{Ty})
    nx = length(bxx)
    ny = length(byy)
    bxy = Matrix{Tuple{Tx,Ty}}(nx,ny)
    for ix=1:nx
        for jy=1:ny
            bxy[ix,jy] = (bxx[ix],byy[jy])
        end
    end
    vec(bxy)
end

# [[x,y],[x',y'], ...] ⊗ [[z],[z'],...]  --> [[x,y,z], [x',y',z], ... , [x,y,z'], ...]
function flat_tensor{Tx}(bxx::Vector{Vector{Tx}},byy::Vector{Vector{Tx}})
    nx = length(bxx)
    ny = length(byy)
    bxy = Matrix{Vector{Tx}}(nx,ny)
    for ix=1:nx
        for jy=1:ny
            bxy[ix,jy] = vcat(bxx[ix],byy[jy])
        end
    end
    vec(bxy)
end




######################################################################################
### Internal API for common Markov operations
######################################################################################


# Markov rule: nu = mu Q
upq(mux,qxy)=vec(mux'*qxy)   #un-optimized

upq!(nuy,mux,qxy)=At_mul_B!(nuy,qxy,mux)

function upj!(nuxy,mux,qxy)
    nx,ny = size(qxy)
    for jy=1:ny
        for ix=1:nx
            nuxy[ix,jy] = mux[ix]*qxy[ix,jy]
        end
    end
end


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



######################################################################################
### Diagnostic function for filtering
######################################################################################

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


######################################################################################
### Manipulate filter output
######################################################################################

function nlf_quantile(nlf,bxx,q,i=1)
    T=size(nlf,2)
    qq=zeros(T)
    xx=[x[i] for x=bxx]
    for t=1:T
        qq[t]=quantile(xx,Weights(nlf[:,t]),q)
    end
    qq
end

function nlf_means(nlf,bxx)
    T  = size(nlf,2)
    xx = mat(bxx)
    dx = size(xx,1)
    m  = zeros(dx,T)
    xx*nlf
end







