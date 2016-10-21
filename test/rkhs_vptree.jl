# import HiddenMarkovModels.evaluate #do I need to import to extend?

immutable Euclid <: Distance end

HiddenMarkovModels.evaluate(distance::Euclid,x,y)=norm(x-y)   #method extension

data=[
0.314781, 0.783293, 0.006127, 0.835617, 0.022911, 
0.670306, 0.763567, 0.338285, 0.875062, 0.661543, 
0.395832, 0.264065, 0.968387, 0.940553, 0.547771, 
0.067000, 0.226725, 0.687411, 0.413454, 0.115242
]
tree=VPTree(data,Euclid())
nn=knn(tree,0.5,5)

# println(data[nn])
# println(nn)
@test sort(nn)==[8,10,11,15,19]



distance_table=[evaluate(Euclid(),data[i],data[j]) for i=1:length(data),j=1:length(data)]
itree=VPTree(data,Euclid(),distance_table)
nn=knn(tree,0.5,5)

@test sort(nn)==[8,10,11,15,19]




