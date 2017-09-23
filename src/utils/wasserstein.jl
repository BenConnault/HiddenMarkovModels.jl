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
