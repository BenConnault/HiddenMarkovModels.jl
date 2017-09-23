

doc"""
    dtv(mu1,mu2)

Total variation distance. Normalized so that ``d_{TV} \leq 1``.
"""
dtv(mu1,mu2)=norm(vec(mu1-mu2),1)/2


doc"""
    dhilbert(mu1,mu2)

Hilbert projective distance between two positive vectors.
"""
function dhilbert(mu1::AbstractVector,mu2::AbstractVector)
    @assert sum(mu1 .<= 0)==0 #check positivity
    @assert sum(mu2 .<= 0)==0
    z=mu1./mu2
    log(maximum(z)/minimum(z))
end

doc"""
    dhilbert(m1,m2)

Hilbert projective distance between two positive definite matrices.
"""
function dhilbert(m1::AbstractMatrix,m2::AbstractMatrix)
    @assert isposdef(m1)==1
    @assert isposdef(m2)==1
    m=m2\m1
    e=eigvals(m)
    log(maximum(e)/minimum(e))
end



