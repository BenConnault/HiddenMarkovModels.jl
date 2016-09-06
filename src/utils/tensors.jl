
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


opnormo(m,::Type{Val{1}})=sum(sqrt.(max(eigvals(m'*m),0)))
opnormo(m,::Type{Val{2}})=norm(vec(m))
opnormo(m,::Type{Val{0}})=sqrt(eigmax(m'*m))
opnormo(m,p=0)=opnormo(m,Val{p})

opnorm(m,p=Inf)=norm(sqrt.(svdvals(m'*m)),p)

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