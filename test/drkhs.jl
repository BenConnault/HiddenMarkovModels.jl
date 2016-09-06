
using DRKHS

n1,n2=3,4
k1=rdrkhs(n1)
k2=rdrkhs(n2)
k12=kron(k1,k2)
u1=sqrtm(k1)
u2=sqrtm(k2)
u12=kron(u1,u2)
m1=mult(k1,u1)
m2=mult(k2,u2)
m12=mult(k12,u12)
j1=incl(k1,u1)
j2=incl(k2,u2)
jt1=inclt(k1,u1)
jt2=inclt(k2,u2)

mu=rsm(1,n1)
f=rand(n1)
f1=rand(n1)
f2=rand(n1)
g=rand(n2)
g2=rand(n2)
g3=rand(n2)
q=rsm(n1,n2)

cmg=diagm(g)   #canonical version
mg0=mg(k2,u2,g)
mtmu0=mtmu(k1,u1,mu)

mt1=mtf(k1,u1,ones(n1))
omt1=u1*mt1*u1'*inv(k1)   #\tilde{M}_1 in pointwise coordinates
mt2=mtf(k2,u2,ones(n2))
mtqmu=mtmu(k2,u2,mu*q)

uf1=u1'*(k1\f1)
ug=u2'*(k2\g)
ug2=u2'*(k2\g2)
fmu0=fmu(k1,mu)
ufmu=u1'*(k1\fmu0)
mtmu1=mtf(k1,u1,fmu0)   #same as mtmu

mqg=mg(k1,u1,q*g) # so we have u1*mqg*[1,0,0] = u1[:,1].*(q*g) = u11(x)Q(x,dy)g(y)
qd=qdual(k1,u1,u2,q)
qc=qchannel(k1,u1,u2,q)


tol=1e-10



@test norm(cmg*u2[:,1]-u2*mg0*ei(1,n2))   < tol
@test norm(fmu0'*(k1\f)-mu*f) < tol
@test norm(mtmu(k1,u1,mu)-mtf(k1,u1,fmu(k1,mu)))   < tol
@test norm(ones(n1)'*(k1\f)-(u1*mt1*u1'*(k1\f))'*(k1\ones(n1)))   < tol   #\ip{\tilde{M}_f g}{h}=\ip{f}{gh}
@test norm(dot(ones(n1),k1\(f1.*f2))-dot(omt1*f1,k1\f2))   < tol  #\ip{\tilde{M}_1 f}{f'}=\ip{1}{ff'}
@test norm(u1*mqg*ei(1,n1)-u1[:,1].*(q*g))   < tol  #\tilde{M}_{Qg} u1
@test norm(dot(mtqmu*ug,ug)-mu*q*(g.^2))   < tol   #\ip{\tilde{M}_Q\mu g}{g}=\ip{\mu}{Qg^2}
# println("TEST ",norm(m12*kron(myst1*uf,ug,ufa,uga)-kron(m1*kron(myst1*uf,ufa),m2*kron(ug,uga))))  #of course that's ok
@test norm(mg(k1,u1,f1)-j1*kron(eye(n1),uf1))   < tol     #test incl()  
@test norm(mtf(k1,u1,f1)-jt1*kron(eye(n1),uf1))   < tol     #test inclt()  
@test norm(u1*qdual(k1,u1,u2,q)*ug-q*g)   < tol     #test qdual()  
@test norm(u2*qchannel(k1,u1,u2,q)*ufmu-fmu(k2,mu*q))   < tol     #test qchannel()  
# @test norm(u2*qchannel(k1,u1,u2,q)*ufmu-fmu(k2,mu*q))   < tol     #test jointq()  

# USE THAT FOR TESTS OF mt2f and ismtf
# println("f ", fa)
# println("f? ",mt2f(k1,u1,mtfa))
# println("is? ",ismtf(k1,u1,mtfa))



ac=quantum(k1,u1,k2,u2,q)
# @test norm(ac'*kron(mtmu0,eye(n2))*ac-mtqmu)  < tol   #TOFIX

# I am not sure how to test dualquantum since by design, sometimes
# it succeeds, sometimes it does not.
# ac2,ad2=dualquantum(k1,u1,k2,u2,q)
# println(norm(stine(ad2,mg)-mqg))
# println(norm(stine(ac2,mtmu)-mtqmu))


