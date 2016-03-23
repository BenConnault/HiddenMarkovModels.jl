module KHMM

using JuMP, Ipopt

import Base.length

line(x)=reshape(x,1,lengh(x))

include("rkhs_types.jl")
include("rkhs_vptree.jl")
# include("rkhs_filtering.jl")

export GaussianRKHS, DiscreteRKHS, Point, RKHSVector, RKHSMap, KernelDistance
export VPTree, knn
export dimension,length

end