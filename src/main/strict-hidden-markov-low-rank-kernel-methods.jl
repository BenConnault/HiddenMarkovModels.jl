
struct LRKKF_SHHM{Tx,Ty} <: LRKXF
    bxx::Vector{Tx}     # B_x basis
    xx0::Vector{Tx}     # subsample carrying Nystrom approximations 
    byy::Vector{Ty}     # B_y basis
    yy0::Vector{Ty}     # subsample carrying Nystrom approximations 

    #qxx ≈ gx20 * [(gx01*gx10)\gx01]  in (B_x,B_x)
    #qxy ≈ gx20 * [(gx01*gx10)\gx01]  in (B_x,B_y)
    gx20::Matrix{Float64}       # gramian(bxx,xx0)
    wx00::Matrix{Float64}       # = gx01*gx10
    gx01::Matrix{Float64}       # gramian(xx1,xx0)
    gy40::Matrix{Float64}       # gramian(byy,yy0)
    wy00::Matrix{Float64}       # = gy04*gy40
    wxy14::Matrix{Float64}      # = gx01*gy40
end


"""
    LRKF(model::StrictHMM,r,m)

Build approximate dynamics for `model`. 
Markov transition matrices are supported on `m` basis points and are of low rank `r`.
Rely on user-provided `draw_x`, `draw_y` and `rand_ux`.
"""
function LRKF(model::StrictHMM,r,m)

    xx1 = Vector{Vector{Float64}}(m)        # "left" sample on E_x (ie. x_t)     
    bxx = Vector{Vector{Float64}}(m)        # "right" sample on E_x (ie. x_t+1), will eventually be bxx. Sometimes called xx2.
    byy = Vector{Vector{Float64}}(m)        # 

    for i=1:m
        xx1[i] = rand_ux(model)
        bxx[i] = rand(model,Val(:x),xx1[i])
        byy[i] = rand(model,Val(:y),xx1[i])
    end

    xx0 = sample(xx1,r,replace=false)  #if necessary, may want to subsample from vcat(bxx,xx1)
    yy0 = sample(byy,r,replace=false)

    gx01 = gramian(xx0,xx1)
    wx00 = gx01*gx01'
    gx20 = gramian(bxx,xx0)
    gy40 = gramian(byy,yy0)
    wy00 = gy40'*gy40
    wxy14 = gx01*gy40


    LRKKF_SHMM(bxx,xx0,byy,yy0,gx20,wx00,gx01,gy40,wy00,wxy14)
end


function mutation!(pred,model::StrictHMM,mu,kf::KKF_SHHM)
    unnormalized_predictive = kf.gx01'*(kf.wx00\(kf.gx20'*mu))
    probnorm!(pred,unnormalized_predictive)
end



function selection!(mu,model::StrictHMM, predictive, y_tp1, kf::KKF_SHHM)
    # nu_y is the unnormalized marginal on E_y
    nu_y = kf.gx01'*(kf.wx00\(kf.gx20'*predictive))
    
    a5 = gramian(kf.yy0,[y_tp1])
    w1 = kf.gy40'*(nu_y .* kf.gy40)
    w2 = kf.gy40'*((nu_y.^2) .* kf.gy40)
    temp = (kf.wx00\(kf.wxy14*(w2\(w1*(kf.wy00\a5)))))
    
    # pi_x is the unnormalized posterior
    pi_x =  (predictive.*kf.gx20)*temp
    probnorm!(mu,pi_x)
end
