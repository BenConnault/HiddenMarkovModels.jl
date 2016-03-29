A = [ 0.99 0.01 ;
      0.01 0.99 ]
B = [ 1/6   1/6   1/6   1/6   1/6   1/6 ;
      1/10  1/10  1/10  1/10  1/10  1/2 ]
p = [0.625, 0.375]

data = [1, 2, 1, 6, 6] # observation sequence

xx=[1,2]
yy=[1,2,3,4,5,6]

model=hmm((A,B),p*fill(1/6,1,6))
fil,smo,cond=filtr(model,data)  # from DynamicDiscreteModels.jl

Hx  = DiscreteRKHS{1}()
Hy  = DiscreteRKHS{2}()
Hxy = (Hx,Hy)
Bx  = RKHSBasis(Hx,xx)
By  = RKHSBasis(Hy,yy)
Bxy = RKHSBasis(Hxy,vec([(ix,jy) for ix=xx,jy=yy]))

mk1=RKHSMap(Bx,A*B,By)   # P(Y' | X)
Q=[A[ix,jx]*B[jx,jy] for ix=1:2,jy=1:6,jx=1:2]
Q=reshape(Q,12,2)
Q=Q./sum(Q,2)
mk2=RKHSMap(Bxy,Q,Bx)   # P(X'| X,Y')
ini=RKHSVector(p,Bx)

filter_from_rkhs=filtr(mk1,mk2,ini,data)

true_filter= [ 0.625   0.7332  0.8173  0.5884  0.3212 ;
                 0.375   0.2668  0.1827  0.4116  0.6788 ]

filter_from_ddm=fil

tol=1e-3

@test norm(vec(filter_from_rkhs   - true_filter     ))   < tol 
@test norm(vec(filter_from_rkhs   - filter_from_ddm ))   < tol 