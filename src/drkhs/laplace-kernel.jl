doc"""
   klaplace(xx,alpha=1.0)

Return the kernel matrix `K` for the discrete RKHS with Laplace kernel `k(x,x')=exp(-alpha*abs(x-x'))` 
and supported on the points `xx`.
"""
klaplace(xx::AbstractVector,alpha=1.0)=[exp(-alpha*abs(xx[i]-xx[j])) for i=1:length(xx),j=1:length(xx)]

doc"""
   klaplace(n,alpha=1.0)

Return the kernel matrix `K` for the discrete RKHS with Laplace kernel `k(x,x')=exp(-alpha*abs(x-x'))` 
and supported on the points `linspace(0,1,n)`.
"""
klaplace(n::Int,alpha=1.0)=klaplace(linspace(0,1,n),alpha)

doc"""
   invklaplace(x,sorted=false)

Return a fast closed-form expression of the inverse of the kernel matrix `K` 
for the discrete RKHS with Laplace kernel `k(x,x')=exp(-alpha*abs(x-x'))` 
and supported on the points `x`. The inverse is always tridiagonal.
"""
function invklaplace(x,sorted=false)
    xx=sorted?x:sort(x)
    n=length(x)
    ik=zeros(n,n)  #consider symtridiagonal of sparse structure
    for i=2:n-1
        rho2im=exp(-2*abs(xx[i]-xx[i-1]))                #rho_{i-1}^2
        rho2i=exp(-2*abs(xx[i+1]-xx[i]))                #rho_i^2
        ik[i,i]=1/(1-rho2i)+rho2im/(1-rho2im)   #a_i
        ik[i+1,i]=-sqrt(rho2i)/(1-rho2i)        #c_i
        ik[i,i+1]=ik[i+1,i]                     #c_i
    end
    rho21=exp(-2*abs(xx[2]-xx[1]))
    ik[1,1]=1/(1-rho21)
    ik[2,1]=-sqrt(rho21)/(1-rho21)
    ik[1,2]=-sqrt(rho21)/(1-rho21)
    ik[n,n]=1/(1-exp(-2*abs(xx[n]-xx[n-1])))
    ik
end

doc"""
   laplacenorm(xx,ff)

Compute the norm of `ff` in the discrete Laplace RKHS supported on the points `xx, using a
fast closed-form expression. `xx` must be sorted.
Call ` laplacenorm(xx,map(f,xx))` if `f` is given as a function.
"""
function laplacenorm(xx,ff)
    #must be sorted
    n=length(xx)
    w=zeros(n)
    w2=zeros(n-1)
    rho=exp.(-abs.(view(xx,2:n)-view(xx,1:n-1)))
    w=[(1-rho[i-1]rho[i])/((1+rho[i-1])*(1+rho[i])) for i=2:n-1]
    w2=rho./(1-rho.^2)
    nn=dot(w,view(ff,2:n-1).^2)+dot(w2,(view(ff,2:n)-view(ff,1:n-1)).^2)
    f1=(1+(1-rho[1])/(1+rho[1]))*ff[1]^2/2
    fn=(1+(1-rho[n-1])/(1+rho[n-1]))*ff[n]^2/2
    nn+f1+fn
end