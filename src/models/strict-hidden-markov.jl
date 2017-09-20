######################################################################################
### Implementation of the AbstractHiddenMarkovModel interface
######################################################################################

abstract type StrictHiddenMarkovModel <: AbstractHiddenMarkovModel
    # transition
    # measurement
end

function rand(model::StrictHiddenMarkovModel,xy::Tuple)
    x,y = xy
    x2  = rand(model.transition,x)
    y2  = rand(model.measurement,x2)
    (x2,y2)
end

######################################################################################
### Implicit interface that all StrictHiddenMarkovModels must implement
######################################################################################

## FIELDS

# `transition` and `measurement` (see also METHODS)

## METHODS

# # Sample x' starting from x 
# rand(model.transition,x)

# # Sample y starting from x 
# rand(model.measurement,x)


######################################################################################
### Methods common to all StrictHiddenMarkovModel
######################################################################################


struct KernelFilterForStrictHiddenMarkovModel <: KernelOrBasis
# struct KernelFilterForStrictHiddenMarkovModel{Tx,Ty} <: KernelOrBasisFilter
    # bx::Vector{Tx}
    # by::Vector{Ty}
    bxx::Vector{Vector{Float64}}
    byy::Vector{Vector{Float64}}
    kx::Matrix{Float64}
    ky::Matrix{Float64}
    qxx::Matrix{Float64}
    qxy::Matrix{Float64}
    m::Int          # number of simulation draws used for each row
    tol::Float64    # regularization parameter
end


function KF(model,bxx,byy,qxx::AbstractMatrix,qxy,m::Int=500,tol=0.0) # is m needed after qxx, qxy are done? I think not
    kx=gramian(bxx)
    ky=gramian(byy)
    KernelFilterForStrictHiddenMarkovModel(bxx,byy,kx,ky,qxx,qxy,m,tol)
end


function KF(model,bxx,byy,m::Int=500,tol=0.0)
    kx=gramian(bxx)
    ky=gramian(byy)

    print("Building transition matrix... ")
    qxx=markovapprox(model.transition,  bxx, bxx, kx, m, tol)
    println()

    print("Building observation matrix... ")
    qxy=markovapprox(model.measurement, bxx, byy, ky, m, tol)
    println()
    
    KernelFilterForStrictHiddenMarkovModel(bxx,byy,kx,ky,qxx,qxy,m,tol)
end


function filter_update(model::StrictHiddenMarkovModel,mu,y_t,y_tp1,kf::KernelFilterForStrictHiddenMarkovModel)
    nx,ny=size(kf.qxy)

    predictive=upq(mu,kf.qxx)    #if comp. bottleneck to create array, can stuff in `kf`

    gy=Array{Float64}(ny)
    for iy=1:ny
        gy[iy]=kk(kf.byy[iy],y_tp1)
    end
    kbr(kf.qxy,kf.ky,predictive,gy,1.0)
end



