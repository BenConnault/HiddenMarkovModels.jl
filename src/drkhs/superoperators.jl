doc"""
    krausq(k1,u1,k2,u2,q)

Compute the kraus operator `kr` such that calling `channel(kr,mtmu)` will compute 
``\Phi_Q(\tilde{M}_{\mu})=\tilde{M}_{\mu Q}``
and calling `dual(kr,m)` will compute ``\Phi_Q'(M)`` (``\Phi_Q'(M_g)`` is not necessarily ``M_{Qg}``) 
"""
function krausq(k1,u1,k2,u2,q)
    n1,n2=size(q)
    smtq=zeros(n2,n2,n1)
    for i1=1:n1
        mtqi=mtmu(k2,u2,q[i1,:]')
        smtq[:,:,i1]=real(sqrtm(mtqi))
    end
    n3=n2
    kr=zeros(n2,n1,n3)
    for i3=1:n3
        aik=[smtq[i2,i3,j1] for i2=1:n2,j1=1:n1]
        kr[:,:,i3]=aik*(k1\u1)
    end
    kr
end




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





# for a quantum channel (aka predual) H1 to H2 
# sending MT (n1 x n1) to Phi(MT) (n2 x n2)
# with dual sending M (n2 x n2) to Phi'(M) (n1 x n1)
# aa (n2 x n1 x n3) such that
# Phi(MT)= sum_i3 aa[:,:,i3]  * MT * aa[:,:,i3]'
# Phi'(M)= sum_i3 aa[:,:,i3]' * M  * aa[:,:,i3]
channel(kraus,mt)=sum(view(kraus,:,:,i3)*mt*view(kraus,:,:,i3)' for i3=1:size(kraus,3))
dual(kraus,m)=sum(view(kraus,:,:,i3)'*m*view(kraus,:,:,i3) for i3=1:size(kraus,3))



#NOT MY FAVORITE WAY OF COMPUTING CHANNEL ANYMORE

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
    the quantum channel (superoperator) induced by `q`, meaning the JOINT
    ``\tilde{M}_{\mu Q}`` can be computed as `ac'*kron(mtmu,eye(n2))*ac`.
    This implies that the superoperator is completely positive.
    """
    quantum(k1,u1,k2,u2,q)=real(sqrtm(bsquare(k1,u1,k2,u2,q)))

#



