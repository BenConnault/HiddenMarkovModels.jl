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
   quantum(k1,u1,k2,u2,q)

Given a transition matrix `q`, return a Stinespring matrices `ac` compatible with
the quantum channel (superoperator) induced by `q`, meaning
``\tilde{M}_{\mu Q}`` can be computed as `ac'*kron(eye(n12),mtmu)*ac`.
This implies that the superoperator is completely positive.
"""
quantum(k1,u1,k2,u2,q)=real(sqrtm(bsquare(k1,u1,k2,u2,q)))





