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
    kkx::Kernel
    kky::Kernel
end

mutation!(pred,model::StrictHMM,mu,kf::KKF_SHHM) = upq!(pred,mu,kf.qxx)    

function selection!(mu,model::StrictHMM, predictive, y_tp1, kf::KKF_SHHM)
    # nx,ny=size(kf.qxy)
    kernel_bayes!(mu,predictive,kf.qxy,y_tp1,kf.byy,kf.ky,kf.kky,kf.tol)
end


function KKF(model::StrictHMM,bxx,byy,qxx,qxy,tol_kbr=1.0,kkx=Laplace(),kky=Laplace()) 
    kx=gramian(bxx,kkx)
    ky=gramian(byy,kky)
    KKF_SHHM(bxx,byy,kx,ky,qxx,qxy,tol_kbr,kkx,kky)
end

rand(model::HiddenMarkovModel,flag::Val{:x},x) = draw_x(model,x)
rand(model::HiddenMarkovModel,flag::Val{:y},x) = draw_y(model,x)

function KKF(model::StrictHMM,bxx,byy,m::Int,tol_kbr=1.0,kkx=Laplace(),kky=Laplace(),tol_approx=0.0)
    kx=gramian(bxx,kkx)
    ky=gramian(byy,kky)

    print("Building transition matrix... ")
    qxx=markovapprox(model, Val(:x), bxx, bxx, kx, m, kkx, tol_approx)
    println()

    print("Building observation matrix... ")
    qxy=markovapprox(model, Val(:y), bxx, byy, ky, m, kky, tol_approx)
    println()
    
    KKF_SHHM(bxx,byy,kx,ky,qxx,qxy,tol_kbr,kkx,kky)
end

function KKF(model::StrictHMM,bxx,byy,tol_kbr=1.0,kkx=Laplace(),kky=Laplace())
    kx=gramian(bxx,kkx)
    ky=gramian(byy,kky)

    print("Building transition matrix (using densities)... ")
    qxx=markovapprox(model, Val(:x), bxx, bxx)
    println()

    print("Building observation matrix (using densities)... ")
    qxy=markovapprox(model, Val(:y), bxx, byy)
    println()
    
    KKF_SHHM(bxx,byy,kx,ky,qxx,qxy,tol_kbr,kkx,kky)
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