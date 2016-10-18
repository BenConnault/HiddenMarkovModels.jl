
doc"""
   ei(i,n)

Return the vector of length ``n`` with 1 in position ``i`` and 0 otherwise.
"""
ei(i,n)=[1.0*(j==i) for j=1:n]


doc"""
    vecpq(p,q,c)

A reshape of the matrix ``C`` such that if ``A`` is ``m \times n``, 
``B`` is ``p \times q``, then `vecpq(p,q,kron(a,b))==vec(a)vec(b)'`.
In particular `rank(vecpq(p,q,c))==1` iff ``C`` can be written 
``A \otimes B`` for suitably-sized matrices ``A`` and ``B``. 
"""
function vecpq(p::Int,q::Int,a::Matrix)
    mp,nq=size(a)
    m=Int(mp/p)
    n=Int(nq/q)
    b=zeros(m*n,p*q)
    for im=1:m
        for jn=1:n
            b[m*(jn-1)+im,:]=reshape(a[(p*(im-1)+1):im*p,(q*(jn-1)+1):jn*q],1,p*q)
        end
    end
    b
end

doc"""
    vecpq(c)

If ``C`` is ``m^2 \times n^2``, call vecpq(m,n,c).
`rank(vecpq(c))==1` iff ``C`` can be written ``A \otimes A``
for some ``A``.
"""
vecpq(a::Matrix)=vecpq(Int(sqrt(size(a,1))),Int(sqrt(size(a,2))),a)



#compute an orthonormal basis for the nullspace of an operator.
# function nullbasis(a)
#     m,n=size(a)
#     ss=svd(a,thin=false)
#     nsn=sum(norm.(ss[2]).> 1e-10)  #number of non-zero singular values (=rank)
#     ss[3][:,nsn+1:n]
# end


# function gramschmidt(vv)
#     v=copy(vv)
#     m,n=size(v)
#     r = eye(n)
#     for i=1:n
#         vi=view(v,:,i)
#         r[i,i+1:n]=vi'*view(v,:,i+1:n)/dot(vi,vi)
#         v[:,i+1:n]=view(v,:,i+1:n)-vi*view(r,i,i+1:n)'
#     end
#     v*inv(sqrtm(v'*v))
# end

# vv=rand(10,4)
# pvv=vv*((vv'*vv)\vv')
# vvv=gramschmidt(vv)
# println("GS1 ",norm(eye(4)-vvv'*vvv))
# println("GS2 ",norm(pvv*vvv-vvv))


doc"""
    odec(x)

For a symmetric matrix X, compute the decomposition ``X=P_1-P_2``
where ``P_1`` and ``P_2`` are psd and ``P_1 P_2=0``. 
"""
function odec(s)
    v,p=eig(s)
    vneg=-v.*(v.<0)
    vpos=v.*(v.>0)
    mneg=p*diagm(vneg)*p'
    mpos=p*diagm(vpos)*p'
    mneg,mpos
end

doc"""
    opnorm(m,p=Inf)

Compute the `p`-Schatten operator norm of `m` via the `p`-norm of its singular values.
"""
opnorm(m,p=Inf)=norm(svdvals(m),p)


#FOR TESTS
# opnorm1(m)=trace(sqrtm(m'*m))
# opnorm2(m)=sqrt(sum(eigvals(m'*m)))

# function ohmy(m,n)
#     mm=randn(m,n)
#     println("1 ", opnorm(mm,1))
#     println("1 ", opnorm1(mm))
#     println("2 ", opnorm(mm,2))
#     println("2 ", opnorm2(mm))
#     println("0 ", opnorm(mm))
#     println("0 ", norm(mm))
# end




doc"""
    partialtrace(m,n1,n2,r)

Compute the partial trace of ``M \in B(H_1 \otimes H_2)`` on ``H_r``.
For example `partialtrace(kron(eye(n),B),n,size(B,1),1)` will return `n*B` 
(summing over the diagonal blocks.)
"""
function partialtrace(m,n1,n2,r)
    m=reshape(m,n2,n1,n2,n1)
    r==2 && return sum(m[i2,:,i2,:] for i2=1:n2)
    r==1 && return sum(m[:,i1,:,i1] for i1=1:n1)
end

doc"""
    partialtrace(m,r)

`partialtrace(m,n1,n2,r)` when `n1=n2`.
"""
function partialtrace(m,r)
    n=Int(sqrt(size(m,1)))
    partialtrace(m,n,n,r)
end


doc"""
    rortho(n)

Return a random orthogonal matrix. Use a naive method rather than the Haar measure.
"""
function rortho(n)
    x=randn(n,n)
    xx=x'x
    sqrtm(xx)\x'
end