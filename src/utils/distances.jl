doc"""
    wasserstein(mu1,mu2,d)

Return the 1-wasserstein coupling and distance `(lambda,d)` 
between the discrete probability measures `mu1` and `mu2`.
`d` is the ``n \times n`` matrix containing the pairwise distances between the support points.
"""
function wasserstein(mu1,mu2,d::AbstractMatrix)
    n=length(mu1)
    @assert length(mu2)==n
    @assert size(d)==(n,n)
    m = Model()
    @variable(m, 0.0 <= lambda[1:n,1:n] <=1.0)
    @constraint(m, sum(lambda[i,j] for i=1:n,j=1:n)==1)     
    @constraint(m, lambda*ones(n) .== mu1)     
    @constraint(m, lambda'*ones(n) .== mu2)     
    @objective(m, Min, sum(lambda[i,j]*d[i,j] for i=1:n,j=1:n))
    status=solve(m)
    getvalue(lambda),getobjectivevalue(m)
end


doc"""
    dwasserstein(mu1,mu2,d)

Return the 1-wasserstein distance between the discrete probability measures `mu1` and `mu2`.
`d` is the ``n \times n`` matrix containing the pairwise distances between the support points.
"""
dwasserstein(mu1,mu2,d::AbstractMatrix)=wasserstein(mu1,mu2,d::AbstractMatrix)[2]


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



