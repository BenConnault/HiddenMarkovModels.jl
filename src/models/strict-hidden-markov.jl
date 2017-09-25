######################################################################################
### Implementation of the HiddenMarkovModel interface
######################################################################################

abstract type StrictHMM <: HiddenMarkovModel
end

function rand(model::StrictHMM,xy::Tuple)
    x,y = xy
    x2  = draw_x(model,x)
    y2  = draw_y(model,x2)
    (x2,y2)
end

######################################################################################
### Implicit interface that all StrictHMMs must implement
######################################################################################

## METHODS
# draw_x(model,x)
# draw_y(model,x2)

######################################################################################
### Methods common to all StrictHMM
######################################################################################

function filter_update!(mu_tp1,predictive,model::StrictHMM,mu_t,y_t,y_tp1,filtering_method)
    mutation!(predictive,model,mu_t,filtering_method) 
    selection!(mu_tp1,model,predictive,y_tp1,filtering_method)
end

######################################################################################
### Kernel Filtering, aka. Kernel-Kernel Filtering, aka. KKF
######################################################################################

struct KKF_SHHM{Tx,Ty} <: KXF
    bxx::Vector{Tx}
    byy::Vector{Ty}
    kx::Matrix{Float64}
    ky::Matrix{Float64}
    qxx::Matrix{Float64}
    qxy::Matrix{Float64}
    tol::Float64    # regularization parameter
end

function KKF(model::StrictHMM,bxx,byy,qxx,qxy,tol_kbr=1.0) 
    kx=gramian(bxx)
    ky=gramian(byy)
    KKF_SHHM(bxx,byy,kx,ky,qxx,qxy,tol_kbr)
end

function KKF(model::StrictHMM,bxx,byy,tol_kbr=1.0,m::Int=500,tol_approx=0.0)
    kx=gramian(bxx)
    ky=gramian(byy)

    print("Building transition matrix... ")
    qxx=markovapprox(model, Val(:x), bxx, bxx, kx, m, tol_approx)
    println()

    print("Building observation matrix... ")
    qxy=markovapprox(model, Val(:y), bxx, byy, ky, m, tol_approx)
    println()
    
    KKF_SHHM(bxx,byy,kx,ky,qxx,qxy,tol_kbr)
end

mutation!(pred,model::StrictHMM,mu,kf::KKF_SHHM) = upq!(pred,mu,kf.qxx)    

function selection!(mu,model::StrictHMM, predictive, y_tp1, kf::KKF_SHHM)
    nx,ny=size(kf.qxy)
    gy=Array{Float64}(ny)
    for iy=1:ny
        gy[iy]=kk(kf.byy[iy],y_tp1)
    end
    mu[:]=kbr(kf.qxy,kf.ky,predictive,gy,kf.tol)
end



######################################################################################
### Kernel-Density Filtering, aka. KDF
######################################################################################

struct KDF_SHHM <: KXF  #TODO
end


mutation!(pred,model::StrictHMM,mu,kf::KDF_SHHM) = upq!(pred,mu,kf.qxx)    


# #TODO
# function selection!(model::StrictHMM, predictive, y_tp1, kf::KDF_SHHM)
# end