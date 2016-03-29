import HiddenMarkovModels.evaluate

immutable Euclid <: Distance end

evaluate(distance::Euclid,x,y)=norm(x-y)

data=float(collect(1:100))
tree=VPTree(data,Euclid())
nn=knn(tree,50.0,5)

@test sort(nn)==[48,49,50,51,52]