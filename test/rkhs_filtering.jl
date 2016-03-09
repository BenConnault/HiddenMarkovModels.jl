A = [ 0.99 0.01 ;
      0.01 0.99 ]
B = [ 1/6   1/6   1/6   1/6   1/6   1/6 ;
      1/10  1/10  1/10  1/10  1/10  1/2 ]
p = [0.625, 0.375]

data = [1, 2, 1, 6, 6] # observation sequence

xx=[1,2]
yy=[1,2,3,4,5,6]

model=hmm((A,B),p*fill(1/6,1,6))
fil,smo,cond=filtr(model,data)

# prediction
# [0.5 0.625 0.7332155477031801 0.47219677537559535 0.23010666673800956
#  0.5 0.375 0.2667844522968198 0.5278032246244047 0.7698933332619905]


mka=RKHSMap(HD{1},HD{1},line(xx),A,xx)
mkb=RKHSMap(HD{1},HD{2},line(xx),B,yy)
mk=chainrule(mka,mkb)
ini=RKHSLeftElement(HD{1},line(p),xx)

pre,up=filtr(mk,ini,data)
rev,post=filtersmoother(mk,ini,data)

true_filter= [ 0.625   0.7332  0.8173  0.5884  0.3212 ;
                 0.375   0.2668  0.1827  0.4116  0.6788 ]

filter_from_rkhs=hcat(map(i->up[i].weights[1],1:5),map(i->up[i].weights[2],1:5))'
filter_from_ddm=fil

smoother_from_rkhs=hcat(map(i->post[i].weights[1],1:5),map(i->post[i].weights[2],1:5))'
smoother_from_ddm=hcat(sum(cond[:,:,1],2),hcat(map(i->sum(cond[:,:,i-1],1)[1],2:5),map(i->sum(cond[:,:,i-1],1)[2],2:5))')

tol=1e-3

@test norm(vec(filter_from_rkhs   - true_filter     ))   < tol 
@test norm(vec(filter_from_rkhs   - filter_from_ddm ))   < tol 
@test norm(vec(smoother_from_rkhs - smoother_from_ddm )) < tol

