doc"""
    purification(mtff)

Canonical purification of the density operator `mtff` on ``H``: a pure state on ``H \otimes H`` whose partial trace is `mtff`. 
"""
function purification(mtff)
    v,p=eig(mtff)
    # k2=kron(k,k)
    # u2=kron(u,u)
    pure=sum(sqrt(v[i])*kron(p[:,i],p[:,i]) for i =1:length(v))
    # mtf(k2,u2,pure)
    pure
end


#NOT EXPORTED
doc"""
   bsquare(k1,u1,k2,u2,q)

For a Markov transition matrix `q` given in canonical coordinates, return `bs` such that
`sqrtm(bs)` is a Stinespring matrix.
"""
function bsquare(k1,u1,k2,u2,q)
    n1,n2=size(q)
    qc=qchannel(k1,u1,u2,q)
    m1=mult(k1,u1)
    k12=kron(k1,k2)
    u12=kron(u1,u2)

    jointq=kron(eye(n1),qc)*m1'
    f1=u1'*(k1\ones(n1))
    mt1=mtf(k1,u1,ones(n1))   
    #NOTE: I actually don't think that mt1 is always psd
    #it does not matter since the result of kron(mtF,eye(n2))\mtf(k12,u12,u12*jointq*F) does not depend on F
    #but it might be less elegant
    
    kron(mt1,eye(n2))\mtf(k12,u12,u12*jointq*f1)
end




doc"""
    dtrace(mt1,mt2)

Trace metric between two density operators (trace-norm of their difference).
"""
dtrace(mt1,mt2)=opnorm(mt1-mt2,1)


doc"""
    dbures(mt1,mt2)

Bures metric between two density operators. 
Equal to the minimal distance between two respective purifications, 
ie the infimum of the norms of the difference of two purifications.
"""
dbures(mt1,mt2)=sqrt(2-2*sqrt(fidelity(mt1,mt2)))

fidelity(mt1,mt2)=opnorm(sqrtm(mt1)*sqrtm(mt2),1)^2    #holevo
# fidelity(mt1,mt2)=trace(sqrtm(sqrtm(mt2)*mt1*sqrtm(mt2)))^2   #alternative expression, do performance comparison








doc"""
   quantum(k1,u1,k2,u2,q)

Given a transition matrix `q`, return a Stinespring matrices `ac` compatible with
the quantum channel (superoperator) induced by `q`, meaning
``\tilde{M}_{\mu Q}`` can be computed as `ac'*kron(mtmu,eye(n2))*ac`.
This implies that the superoperator is completely positive.
"""
quantum(k1,u1,k2,u2,q)=real(sqrtm(bsquare(k1,u1,k2,u2,q)))





