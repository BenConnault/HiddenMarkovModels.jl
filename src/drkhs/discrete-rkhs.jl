
doc"""
    drkhs(n,rho=.5)

Return the kernel function with 1 on the diagonal, ``\rho`` on the subdiagonals and 0 elsewhere, 
meant to be used as an example of a noncanonical discrete RKHS.  
"""
function drkhs(n,rho=.5)
    k=eye(n,n)
    for i=1:n-1
        k[i+1,i]=rho
        k[i,i+1]=rho
    end
    k
end


doc"""
    rdrkhs(n)

Return a random kernel function with 1 on the diagonal.
"""
function rdrkhs(n)
    k=rand(n,n)
    k=k*k'
    s=diagm(1./(sqrt.(diag(k))))
    s*k*s
end


# ip{f}{f'} in H is given by f*w*f', w=inv(k)
# u=sqrtm(Symmetric(k))    #orthonormal basis in H, with u[i,j]= u_j(i)

# A basis for T_1 \otimes T_2 is (u_1 \otimes v_1), (u_1 \otimes v_2),.. (in this order).
# The "inner" index (which moves more frequently) is j on v_j. This is the opposite of vec(m[i,j]).


#Here u*m*kron(u'*w*g1,u'*w*g2)=g1.*g2
# and m*kron(u'*w*g,[1,0,0,0]) = mg*[1,0,0,0]
doc"""
    mult(k,u)

For the RKHS ``k``, return the multiplication operator in ``u`` coordinates. 
`mult` is a operator in ``B(H \otimes H,H)``, ie. a matrix of size (``n \times n^2``).
By definition `u*m*kron(u'*w*g1,u'*w*g2)==g1.*g2`.
"""
function mult(k,u)
    n=size(k,1)
    m=zeros(n,n,n)
    for i=1:n
        for j=1:n
            m[:,j,i]=u'*(k\(u[:,i].*u[:,j]))
        end
    end
    reshape(m,n,n^2)
end



#Here cmg*u[:,1]=g.*u[:,1]=u2*mg*[1,0,0,0]
doc"""
    mg(k,u,g)

For a function ``g`` in the RKHS ``k``, return the operator ``M_g`` in ``u`` coordinates.
``g`` is expressed pointwise, ie. g[j]=g(j).
If ``f`` is expressed in ``u``, ``M_g f`` is ``fg`` expressed in ``u``. 
"""
mg(k,u,g)=u'*(k\diagm(g))*u

doc"""
    mtf(k,u,f)

For a function ``f`` in the RKHS ``k``, return the operator ``\tilde{M}_f`` in ``u`` coordinates. 
``f`` is expressed pointwise, ie. f[i]=f(i).
"""
function mtf(k,u,f)
    n=size(k,1)
    mtf=zeros(n,n)
    for j=1:n
        for i=1:n
            mtf[i,j]=(f'*(k\(u[:,i].*u[:,j])))[1]
        end
    end
    mtf
end


# `mtmu` to `mu` in canonical coordinates
# function mtmu2mu(k,u,mtmu)
    #this needs to be rewritten since I have "un-vec()ed" the definition of inclt
#     n=size(k,1)
#     jt=inclt(k,u)
#     pjt=(jt'*jt)\jt'
#     ufmu=pjt*vec(mtmu)   #ufmu is f_\mu expressed in u-coordinates
#     [dot(ufmu,u*(k\ei(i,n))) for i=1:n]
# end

doc"""
    fmu(k,mu)

Find the representant of the measure ``mu`` in the RKHS ``k``. 
``mu`` and ``fmu(k,mu)`` are expressed pointwise.
"""
fmu(mu,k)=vec(k*mu')


#the following assumes "mu" is a measure, and is equal to m-tilde(f_mu)
#it is NOT m-tilde(mu), mu seen as a function
#Of course here trace(mtmu(mu,k)*mg(k,f)) = mu*f


doc"""
    mtmu(k,u,mu)

For a measure ``mu`` in the RKHS ``k``, return the operator ``\tilde{M}_\mu`` in ``u`` coordinates.
``mu`` is expressed pointwise.
This is the same as computing mtf(k,u,fmu(k,mu)).
"""
function mtmu(k,u,mu)
    u=sqrtm(k)
    n=size(k,1)
    w=inv(k)
    mtmu=zeros(n,n)
    for j=1:n
        for i=1:n
            mtmu[i,j]=(mu*(u[:,i].*u[:,j]))[1]
        end
    end
    mtmu
end




doc"""
    incl(k,u)

For a RKHS ``k``, return the operator ``f \to M_f`` in ``u`` coordinates.
By definition `mg(k,u,f)==incl(k,u)*kron(eye(n),u'*(k\f))`.
"""
function incl(k,u)
    n=size(k,1)
    j=zeros(n^2,n)
    for i=1:n
        j[:,i]=vec(mg(k,u,u[:,i]))
    end
    reshape(permutedims(reshape(j,n,n,n),[1,3,2]),n,n^2)
end


#Note: 
# - this is in fact m-tilde: ie `inclt(k,u)*kron(uf2,uf1)` is equal to ``\tilde{M}(f1,f2)``.
# - this does not correspond to an operator in the infinite-dimensional setting.
# - if you want to obtain the operator f \to \tilde{M}_f \in H_1 \otimes H_1, then this is just M'.
doc"""
    inclt(k,u)

For a RKHS ``k``, return the operator ``f \to vec( \tilde{M}_f)`` in ``u`` coordinates.
By definition `mtf(k,u,f)==inclt(k,u)*kron(eye(n),u'*(k\f))`.
"""
function inclt(k,u)
    n=size(k,1)
    j=zeros(n^2,n)
    for i=1:n
        j[:,i]=vec(mtf(k,u,u[:,i]))
    end
    reshape(permutedims(reshape(j,n,n,n),[1,3,2]),n,n^2)
end



function mt2f(k,u,mtff)
    m=mult(k,u)
    uf=(m*m')\(m*vec(mtff))
    u*uf
end

function ismtf(k,u,mtff,tol=1e-10)
    f=mt2f(k,u,mtff)
    norm(mtff-mtf(k,u,f)) < tol
end




# CODE FOR PROJECTING mg's to g's etc.

# function m2mt(k,u)
#     n=size(k,1)
#     jt=inclt(k,u) # vec(mtf)=jt*uf  (n^2 x n)
#     j=incl(k,u)
#     invj=(j'*j)\j'
#     rjt=reshape(permutedims(reshape(jt,n,n,n),[1,3,2]),n,n^2)
#     a1m2mt=rjt*kron(eye(n),invj)
#     a2m2mt=kron(eye(n),vec(eye(n)))
#     a1m2mt,a2m2mt
# end

# function mt2m(k,u)
#     n=size(k,1)
#     j=incl(k,u)
#     jt=inclt(k,u) # vec(mtf)=jt*uf  (n^2 x n)
#     invjt=(jt'*jt)\jt'
#     rj=reshape(permutedims(reshape(j,n,n,n),[1,3,2]),n,n^2)
#     a1mt2m=rj*kron(eye(n),invjt)
#     a2mt2m=kron(eye(n),vec(eye(n)))
#     a1mt2m,a2mt2m
# end

# a1m2mt,a2m2mt=m2mt(k,u)
# a1mt2m,a2mt2m=mt2m(k,u)

# f=rand(n)
# mff=mg(k,u,f)
# mtff=mtf(k,u,f)
# println(norm(mff-a1mt2m*kron(eye(n^2),mtff)*a2mt2m))
# println(norm(mtff-a1m2mt*kron(eye(n^2),mff)*a2m2mt))




doc"""
    qdual(k1,u1,u2,q)

Return the operator expression of ``g(y) \in H_{k_2} \to Q(x,dy)g(y) \in H_{k_1}`` 
between RKHSs `k1` and `k2` with bases `u1` and `u2`.    
"""
qdual(k1,u1,u2,q)=u1'*(k1\q)*u2  

doc"""
    qchannel(k1,u1,u2,q)

Return the operator which sends ``f_\mu`` in `k1` to ``g_{\mu Q}`` in `k2`, 
expressed in `u1` and `u2` coordinates.    
"""
qchannel(k1,u1,u2,q)=qdual(k1,u1,u2,q)'


doc"""
    jointq(k1,u1,u2,q)

For a Markov transition matrix `Q` (in canonical coordinates), 
return the "joint" operator which sends ``f_\mu`` in ``H_1`` to ``h_{Q \mu}`` in ``H_1 \otimes H_2``, 
expressed in `u1` and `u2` coordinates.
"""
jointq(k1,u1,u2,q)=kron(eye(size(k1,1)),qchannel(k1,u1,u2,q))*mult(k1,u1)'





