tol=1e-10

n1,n2=3,4
n12=n1*n2
a=rand(n1,n1)
b=rand(n2,n2)
c=rand(n12,n12)
@test norm(partialtrace(kron(eye(n1),b),n1,n2,1) - b*n1    < tol
@test norm(partialtrace(kron(eye(n2),b),1) - b*n2    < tol
@test norm(trace(partialtrace(c,n1,n2,2)*a) - trace(c*kron(a,eye(n2))))   < tol


m=rand(8,7)
@test norm(opnorm(m,1)-trace(sqrtm(m'*m))) < tol
@test norm(opnorm(m,2)-sqrt(sum(eigvals(m'*m)))) < tol
