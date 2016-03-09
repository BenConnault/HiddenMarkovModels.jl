p=[.2 .3 .5]
f=[1.0,-1,5]
q=[.6 .4;.7 .3; .2 .8]
g=[.2,1.6]
pp=[.1 .2; .3 .2; .1 .1]


mup=RKHSLeftElement(HD{1},p,[1,2,3])
muf=RKHSRightElement(HD{1},[1 2 3],f)
muq=RKHSMap(HD{1},HD{2},[1 2 3],q,[1,2])
mug=RKHSRightElement(HD{2},[1 2],g)
mupp1=RKHSLeftElement(RKHS2{HD{1},HD{2}},reshape(pp,1,6),vec([(i,j) for i=1:3,j=1:2]))
mupp2=RKHSMap(mupp1)
mupp3=compact(mupp2)
mupp4=RKHSMap(HD{1},HD{2},[1 2 3],pp,[1,2])


pf=sumrule(mup,muf)
pq=sumrule(mup,muq)
pqg1=sumrule(pq,mug)
pqg2=sumrule(mup,muq,mug)
c3=conditioningrule(mupp3)
c4=conditioningrule(mupp4)

cc=chainrule(mup,muq)
b=bayesrule(mup,muq)

@test pf             ≈  p*f
@test pq.weights     ≈  p*q
@test pqg1           ≈  p*q*g
@test pqg2           ≈  p*q*g
@test c3.leftpoints  == c4.leftpoints
@test c3.weights     == c4.weights
@test c3.rightpoints == c4.rightpoints
@test c3.weights     ≈  pp./sum(pp,2)
@test c4.weights     ≈  pp./sum(pp,2)
@test cc.weights     ≈  vec(p).*q
@test b.weights      ≈  transpose(vec(p).*q)./sum(transpose(vec(p).*q),2)

