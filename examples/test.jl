reload("HiddenMarkovModels")

module dev

using HiddenMarkovModels
using Base.Test

A = [ 0.99 0.01 ;
      0.01 0.99 ]
B = [ 1/6   1/6   1/6   1/6   1/6   1/6 ;
      1/10  1/10  1/10  1/10  1/10  1/2 ]
p = [0.625, 0.375]

data = [1, 2, 1, 6, 6] # observation sequence

# compute from external source for comparison
model=hmm((A,B),p*fill(1/6,1,6))
fil,smo,cond=filtr(model,data)  #from DynamicDiscreteModels.jl via HiddenMarkovModels.jl


#HiddenMarkovModels's RKHS filter
Hx=HD{1}
Hy=HD{2}
Hxy=RKHS2{Hx,Hy}
Bx=[1,2]
By=[1,2,3,4,5,6]
Bxy=vec([(i,j) for i=1:2,j=1:6])

mk1=RKHSMap(Hx,Hy,line(Bx),A*B,By)               # P(Y_{t+1}|X_t) expressed in (Bx -> By) 
Q2=reshape([A[ix,jx]*B[jx,jy] for ix=1:2,jy=1:6,jx=1:2],12,2)
Q2=Q2./sum(Q2,2)
mk2=RKHSMap(Hxy,Hx,line(Bxy),Q2,Bx)     # P(X_{t+1}|X_t,Y_{t+1}) expressed in (Bxy -> Bx)
Q3=reshape([A[ix,jx]*B[jx,jy] for ix=1:2,jx=1:2,jy=1:6],2,12)
mk3=RKHSMap(Hx,Hxy,line(Bx),Q3,Bxy)      #P(X_t+1,Y_t+1|X_t)  expressed in (Bx -> Bxy)

ini=RKHSLeftElement(Hx,line(p),Bx)

# filt=filtr(mk,ini,data)
filt,smoo=filtersmoother(mk1,mk2,mk3,ini,data)

# eweights=estep(mk,ini,data)


true_filter= [ 0.625   0.7332  0.8173  0.5884  0.3212 ;
                 0.375   0.2668  0.1827  0.4116  0.6788 ]

filter_from_rkhs=filt
filter_from_ddm=fil

smoother_from_rkhs=smoo
smoother_from_ddm=hcat(sum(cond[:,:,1],2),hcat(map(i->sum(cond[:,:,i-1],1)[1],2:5),map(i->sum(cond[:,:,i-1],1)[2],2:5))')

# cond_from_rkhs=[eweights[k].weights[i,j] for i=1:2,j=1:2,k=1:4]
# cond_from_ddm=cond[:,:,1:4]

tol=1e-3

@test norm(vec(filter_from_rkhs   - true_filter     ))   < tol 
@test norm(vec(filter_from_rkhs   - filter_from_ddm ))   < tol 
@test norm(vec(smoother_from_rkhs - smoother_from_ddm )) < tol
# @test norm(vec(cond_from_rkhs - cond_from_ddm )) < tol



end