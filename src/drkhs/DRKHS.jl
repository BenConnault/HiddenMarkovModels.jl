module DRKHS

importall JuMP, SCS


export drkhs, rdrkhs, mult, mg, mtf, fmu, mtmu, incl, inclt, qdual, qchannel, jointq
include("discrete-rkhs.jl")

export bsquare, quantum
include("superoperators.jl")



end