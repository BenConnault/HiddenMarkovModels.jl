######################################################################################
### Implementation of the AbstractHiddenMarkovModel interface
######################################################################################

abstract type ARHiddenMarkov <: AbstractHiddenMarkovModel
end

function rand(model::ARHiddenMarkov,xy::Tuple)
    x,y = xy
    x2  = draw_x(model,x)
    y2  = draw_y(model,x2,y)
    (x2,y2)
end

######################################################################################
### Implicit interface that all ARHiddenMarkov concrete types must implement
######################################################################################

## METHODS
# draw_x(model,x)
# draw_y(model,x2,y)


######################################################################################
### Methods common to all ARHiddenMarkov
######################################################################################


struct KKF_AHM <: KXF
# struct KKF_AHM{Tx,Ty} <: KernelOrBasisFilter
    # bx::Vector{Tx}
    # by::Vector{Ty}
    bxx::Vector{Vector{Float64}}
    byy::Vector{Vector{Float64}}
    kx::Matrix{Float64}
    ky::Matrix{Float64}
    qxx::Matrix{Float64}
    qxyy::Array{Float64,3}
    m::Int          # number of simulation draws used for each row
    tol::Float64    # regularization parameter for kernel Bayes rule
end


# function KF(model,bxx,byy,qxx::AbstractMatrix,qxy,m::Int=500,tol=0.0) # is m needed after qxx, qxy are done? I think not
function KKF(model::ARHiddenMarkov,bxx,byy,qxx,qxyy,tol_kbr=1.0) # is m needed after qxx, qxy are done? I think not
    kx=gramian(bxx)
    ky=gramian(byy)
    KKF_AHM(bxx,byy,kx,ky,qxx,qxyy,m,tol_kbr)
end


function KKF(model::ARHiddenMarkov,bxx,byy,tol_kbr=1.0,m::Int=500,tol_approx=0.0)
    kx=gramian(bxx)
    ky=gramian(byy)

    print("Building transition matrix... ")
    qxx=markovapprox(model,Val(:x),bxx, bxx, kx, m, tol_approx)
    println()

    print("Building observation matrix... ")
    qxy=markovapprox(model,Val(:y) bxx, byy, ky, m, tol_approx)
    println()
    
    KKF_AHM(bxx,byy,kx,ky,qxx,qxy,m,tol_kbr)
end


struct KDF_AHM <: KXF  #TODO
end



function filter_update(model::ARHiddenMarkov,mu,y_t,y_tp1,filtering_method)
    predictive = mutation(model,mu,filtering_method) 
    selection(model,predictive,y_t,y_tp1,filtering_method)
end


# use union type? loss of performance?
mutation(model::ARHiddenMarkov,mu,kf::KKF_AHM) = upq(mu,kf.qxx)    
mutation(model::ARHiddenMarkov,mu,kf::KDF_AHM) = upq(mu,kf.qxx)    


#TODO
function selection(model::ARHiddenMarkov, predictive, y_tp1, kf::KKF_AHM)
    nx,ny=size(kf.qxy)
    gy=Array{Float64}(ny)
    for iy=1:ny
        gy[iy]=kk(kf.byy[iy],y_tp1)
    end
    kbr(kf.qxy,kf.ky,predictive,gy,kf.tol)
end

# #TODO
# function selection(model::ARHiddenMarkov, predictive, y_tp1, kf::KDF_AHM)
# end