module DRKHS

importall JuMP, SCS


export drkhs, mult, mg, mtf, fmu, mtmu, incl, inclt, qdual, qchannel
include("discrete-rkhs.jl")

export quantum, stinetranspose, stine
include("superoperators.jl")



end