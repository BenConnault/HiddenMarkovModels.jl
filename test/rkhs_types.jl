
m=10

H1=GaussianRKHS{2,1.0,1}()
b1points=[rand(2) for i=1:m]
b1=RKHSBasis(H1,b1points)
H2=DiscreteRKHS{1}()
b2points=[rand(1:10) for i=1:m]
b2=RKHSBasis(H2,b2points)
H3=(H1,H2)
b3points=[(rand(2),rand(1:10)) for i=1:m]
b3=RKHSBasis(H3,b3points)

data=b3.points

tree1=VPTree(data,KernelDistance(H3))

tree2=RKHSBasisTree(b3)


v3=RKHSVector(rand(10),b3)