####### Multilinear algebra routines

###############
## This has become less useful as of 08/2016, since I know now
# (1) A Stinespring representation of the dual map does not necessarily exist.
# (2) A Stinespring representation of the quantum map is available through `I_1 \otimes \tilde{M}_\mu`
#     rather than `I_1 \otimes I_2 \otimes \tilde{M}_\mu`
# (3) I have a more direct way of computing `bs`
## The `dualquantum` function may still be useful to compute counterexamples. 
###############





#a[i,j,k] such that the Stinespring matrix for the dual is A[:,:,1] \\ A[:,:,2] \\ ...
#and the Stinespring matrix for the channel is A[:,:,1]' \\ A[:,:,2]' \\ ...

# `astack(a)` is a reorder of `a` such that astack*astack'=bs (the operator which we will square root)
# `astack(a)` is (n12 x n12)
# the ith column of `astack(a)` is vec(A_i).  
function astack(a)
    n2,n1=size(a)
    n12=n1*n2
    astack=zeros(n12,n12)
    for i=1:n12
        astack[:,i]=vec(slice(a,:,:,i))
    end
    astack
end



# A particular reorder of `a` giving the channel Stinespring matrix from `a`. 
# ``\tilde{M}_{\mu Q}`` can be computed as `ac'*kron(eye(n12),mtmu)*ac`.
achannel(a)=vcat([ctranspose(a[:,:,i]) for i=1:size(a,3)]...)

# A particular reorder of `a` giving the dual Stinespring matrix `ad` from `a`. 
# ``M_{Qg}`` can be computed as `ad'*kron(eye(n12),mg)*ad`.
adual(a)=vcat([a[:,:,i] for i=1:size(a,3)]...)


# Compute the matrix `bc` such that: 
# ``vec(\tilde{M}_{\mu Q})`` can be computed as `bc*vec(mtmu)`.
bchannel(a)=sum([kron(conj(a[:,:,i]),a[:,:,i]) for i=1:size(a,3)])

# Compute the matrix `bd` such that: 
# ``vec(M_{Qg})`` can be computed as `bd*vec(mg)`.
bdual(a)=ctranspose(bchannel(a))

# Compute `bs` from `a`.
# `bs` is a square reshape of `bd` which can be square rooted to obtain (observationally equivalent) `a`. 
function bsquare(a)
    as=astack(a)
    as*transpose(as)
end

# Compute (observationally equivalent) `a` from `bs`.
bs2a(bs,n1,n2)=reshape(sqrtm(bs),n2,n1,n1*n2)

# Compute `bs` from `bd`.
# `bs` is a square reshape of `bd` which can be square rooted to obtain (observationally equivalent) `a`. 
"""doc
    bd2bs(bd,n1,n2)
"""
bd2bs(bd,n1,n2)=reshape(permutedims(reshape(bd,n1,n1,n2,n2),[3,1,4,2]),n1*n2,n1*n2)

# compute the permutation matrix `pi` such that: vec(bd)==pi*vec(bs)
function genpi(n1,n2)
    n12=n1*n2
    pi=zeros(Int,n12^2,n12^2)
    for i1a=1:n1
        for i1b=1:n1
            for i2a=1:n2
                for i2b=1:n2
                    i=sub2ind((n1,n1,n2,n2),i1a,i1b,i2a,i2b)
                    j=sub2ind((n2,n1,n2,n1),i2a,i1a,i2b,i1b)
                    pi[i,j]=1
                end
            end
        end
    end
    pi
end




doc"""
   stinetranspose(a,n1)

Compute an operator ``B`` in ``B(H_2,H_3 \otimes H_1)`` (a matrix of size ``n_3 n_1 \times n_2``) 
from an operator ``A`` in ``B(H_1,H_3 \otimes H_2)`` (a matrix of size ``n_3 n_2 \times n_1``)
such that if ``A' (I_3 \otimes M) A = \sum_{i_3} A_{i_3}' M_{i_3} A_{i_3}``
then ``B' (I_3 \otimes T) B = \sum_{i_3} A_{i_3} T_{i_3} A_{i_3}'``.
"""
function stinetranspose(a,n1)
    n3n1,n2=size(a)
    n3=Int(n3n1/n1)
    reshape(permutedims(reshape(a,n1,n3,n2),[3,2,1]),n3*n2,n1)
end


# doc"""
#    dualquantum(k1,u1,k2,u2,q)

# Given a transition matrix `q`, attempt to return a pair of Stinespring matrices `(ac,ad)` 
# compatible with both the quantum channel induced by `q` and its dual, 
# via a numerical procedure based on SDP programming. Warning:
# There is no theoretical guarantee that such a pair exists, because the dual superoperator can
# fail to be completely positive (use `quantum(k1,u1,k2,u2,q)` if you want a guaranteed Stinespring
# matrix `ac` for the channel side).  
# If successful, this means ``M_{Qg}`` can be computed as `ad'*kron(eye(n12),mg)*ad` and
# ``\tilde{M}_{\mu Q}`` can be computed as `ac'*kron(eye(n12),mtmu)*ac`.
# This implies that the two corresponding superoperators are completely positive.
# """
function dualquantum(k1,u1,k2,u2,q)
    j1=incl(k1,u1)
    j2=incl(k2,u2)
    jt1=inclt(k1,u1)
    jt2=inclt(k2,u2)
    qd=qdual(k1,u1,u2,q)
    qc=qchannel(k1,u1,u2,q)
    n1=size(k1,1)
    n2=size(k2,1)
    n12=n1*n2
    
    # Numerical approach to find a full dual quantum set-up via SDP programming
        # This does not always succeed: more precisely in experiments it looks like 
        # it can succeed or not, both with positive probability  

        #We two types of linear constraints: (1) from the dual side (2) from the channel side 
        #+ the psd constraint enforced by JuMP
        nequations=n1^2*n2+n1*n2^2
        nunknowns=n1^2*n2^2
        aa=vcat(kron(j2',eye(n1^2)),kron(eye(n2^2),jt1'))*genpi(n1,n2)
        @assert size(aa)==(nequations,nunknowns)
        bb=vcat(vec(j1*qd),vec(qc'*jt2'))
        @assert size(bb)==(nequations,)

        prog = Model(solver=SCSSolver(verbose=0))
        @variable(prog, bsx[1:n12,1:n12], SDP )
        @constraint(prog, aa*vec(bsx) .== bb)
        @objective(prog, Min,  1 )
        status = solve(prog)
        bs=Hermitian(getvalue(bsx))
        # println("bs hermitian? ", norm(bs-ctranspose(bs)))
        # println("solved? ", norm(aa*vec(bs)-bb))
    ##

    # Numerical approach for the dual side only
        # aa=kron(j2',eye(n1^2))*genpi(n1,n2)
        # bb=vec(j1*qd)
        # prog = Model(solver=SCSSolver(verbose=0))
        # @variable(prog, bsx[1:n12,1:n12], SDP )
        # @constraint(prog, aa*vec(bsx) .== bb)
        # @objective(prog, Min,  1 )
        # status = solve(prog)
        # zdbs=Hermitian(getvalue(bsx))
    ##

    a=bs2a(bs,n1,n2)
    bd2=bdual(a)
    bc2=bchannel(a)
    # println("still solved? ", norm(aa*vec(bd2)-bb))
    # println("dual constraint ", norm(j1*qd-bd2*j2))
    # println("channel constraint ", norm(jt2*qc-bc2*jt1))
    achannel(a),adual(a)
end