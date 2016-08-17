
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
