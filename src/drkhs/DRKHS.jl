module DRKHS

importall JuMP, SCS

import HiddenMarkovModels.opnorm

export drkhs, rdrkhs, mult, mg, mtf, fmu, mtmu, incl, inclt, mt2f, ismtf,
        qdual, qchannel, jointq
include("discrete-rkhs.jl")


include("laplace-kernel.jl")


export quantum, purification
include("superoperators.jl")



end