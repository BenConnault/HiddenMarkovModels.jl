######################################################################################
### Implementation of the AbstractHiddenMarkovModel interface
######################################################################################

abstract type StrictHiddenMarkov <: AbstractHiddenMarkovModel
end

function rand(model::StrictHiddenMarkov,xy::Tuple)
    x,y = xy
    x2  = draw_x(model,x)
    y2  = draw_y(model,x2)
    (x2,y2)
end

######################################################################################
### Implicit interface that all StrictHiddenMarkovs must implement
######################################################################################

## METHODS
# draw_x(model,x)
# draw_y(model,x2)

######################################################################################
### Methods common to all StrictHiddenMarkov
######################################################################################

# function filter_update(model::StrictHiddenMarkov,mu,y_t,y_tp1,filtering_method)
#     predictive = mutation(model,mu,filtering_method) 
#     selection(model,predictive,y_tp1,filtering_method)
# end

function filter_update!(mu_tp1,predictive,model::StrictHiddenMarkov,mu_t,y_t,y_tp1,filtering_method)
    mutation!(predictive,model,mu_t,filtering_method) 
    selection!(mu_tp1,model,predictive,y_tp1,filtering_method)
end


######################################################################################
### Concrete Discrete Type
######################################################################################

struct DiscreteStrictHiddenMarkov <: StrictHiddenMarkov
    qxx::Matrix{Float64}
    qxy::Matrix{Float64}
end


draw_x(model::DiscreteStrictHiddenMarkov,x) = wsample(model.qxx[x,:])
draw_y(model::DiscreteStrictHiddenMarkov,x) = wsample(model.qxy[y,:])

# mutation(model::StrictHiddenMarkov,mu,kf::DiscreteFilter)  = upq(mu,model.qxx)    
# selection(model::StrictHiddenMarkov, pred, y_tp1, kf::DiscreteFilter)=upf(pred,model.qxy[:,y_tp1])


struct DiscreteFilter <: FilteringTechnique
end


mutation!(pred,model::StrictHiddenMarkov,mu,kf::DiscreteFilter)  = upq!(pred,mu,model.qxx)    
selection!(mu,model::StrictHiddenMarkov, pred, y_tp1, kf::DiscreteFilter)=upf!(mu,pred,view(model.qxy,:,y_tp1))



initial_filter(model,ini,filtering_method::DiscreteFilter) = ini
filtr(model::DiscreteStrictHiddenMarkov,data,ini) = filtr(model,data,ini,DiscreteFilter())



######################################################################################
### Kernel Filtering, aka. Kernel-Kernel Filtering, aka. KKF
######################################################################################

struct KKF_SHM{Tx,Ty} <: KXF
# struct KKF_SHM{Tx,Ty} <: KernelOrBasisFilter
    # bx::Vector{Tx}
    # by::Vector{Ty}
    bxx::Vector{Tx}
    byy::Vector{Ty}
    kx::Matrix{Float64}
    ky::Matrix{Float64}
    qxx::Matrix{Float64}
    qxy::Matrix{Float64}
    # m::Int          # number of simulation draws used for each row
    tol::Float64    # regularization parameter
end


function KKF(model::StrictHiddenMarkov,bxx,byy,qxx,qxy,tol_kbr=1.0) 
    kx=gramian(bxx)
    ky=gramian(byy)
    # KKF_SHM(bxx,byy,kx,ky,qxx,qxy,m,tol_kbr)
    KKF_SHM(bxx,byy,kx,ky,qxx,qxy,tol_kbr)
end


function KKF(model::StrictHiddenMarkov,bxx,byy,tol_kbr=1.0,m::Int=500,tol_approx=0.0)
    kx=gramian(bxx)
    ky=gramian(byy)

    print("Building transition matrix... ")
    qxx=markovapprox(model, Val(:x), bxx, bxx, kx, m, tol_approx)
    println()

    print("Building observation matrix... ")
    qxy=markovapprox(model, Val(:y), bxx, byy, ky, m, tol_approx)
    println()
    
    # KKF_SHM(bxx,byy,kx,ky,qxx,qxy,m,tol_kbr)
    KKF_SHM(bxx,byy,kx,ky,qxx,qxy,tol_kbr)
end

# mutation(model::StrictHiddenMarkov,mu,kf::KKF_SHM) = upq(mu,kf.qxx)    

# function selection(model::StrictHiddenMarkov, predictive, y_tp1, kf::KKF_SHM)
#     nx,ny=size(kf.qxy)
#     gy=Array{Float64}(ny)
#     for iy=1:ny
#         gy[iy]=kk(kf.byy[iy],y_tp1)
#     end
#     kbr(kf.qxy,kf.ky,predictive,gy,kf.tol)
# end


mutation!(pred,model::StrictHiddenMarkov,mu,kf::KKF_SHM) = upq!(pred,mu,kf.qxx)    

function selection!(mu,model::StrictHiddenMarkov, predictive, y_tp1, kf::KKF_SHM)
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

struct KDF_SHM <: KXF  #TODO
end

# mutation(model::StrictHiddenMarkov,mu,kf::KDF_SHM) = upq(mu,kf.qxx)    

mutation!(pred,model::StrictHiddenMarkov,mu,kf::KDF_SHM) = upq!(pred,mu,kf.qxx)    


# #TODO
# function selection(model::StrictHiddenMarkov, predictive, y_tp1, kf::KDF_SHM)
# end