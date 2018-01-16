reload("HiddenMarkovModels")

module dev

using HiddenMarkovModels
HMM = HiddenMarkovModels


import HiddenMarkovModels: draw_x, draw_y, rand_ux

using Plots
plotly()

######################################################################################
### Define a toy model to play with
######################################################################################

# NLG: "Nonlinear" Gaussian
struct NLG <: HMM.StrictHMM
    axx::Matrix{Float64}
    axy::Matrix{Float64}
    sqvxx::Matrix{Float64}
    sqvxy::Matrix{Float64}
    squvxx::Matrix{Float64}
    squvxy::Matrix{Float64}
end


function nlg(d,sigma = 0.6)
    p0=[
      0.51   1.76  -0.49   1.58  -0.21  -0.1   -0.21   0.76  -0.25  -0.58;
     -0.09   0.29  -0.76  -1.34  -1.37   0.24   0.33  -0.7    0.95   0.76;
      0.11  -0.76   0.54   0.72  -0.88  -0.53  -0.05   0.64  -0.03  -0.19;
     -0.95  -1.24  -1.53  -1.99   2.44  -0.36   0.63  -0.31   0.05  -0.1 ;
      1.08  -0.85  -0.23   1.55   0.43  -0.29  -0.26  -0.6   -0.74  -0.11;
     -0.04   0.88   1.68  -0.17   0.35   0.75  -0.03  -2.0    0.78  -0.41;
      1.4    0.98   1.75   0.05  -0.05   0.15   0.98   0.34   0.73  -1.08;
     -1.41   0.1    0.99  -0.34   1.12  -0.04  -1.99   0.66  -0.68  -1.18;
     -1.93   0.72  -0.72   0.12   0.67  -2.73   1.02  -0.51  -0.92  -1.64;
     -1.03  -1.27   0.35  -1.21   0.3   -1.18   0.6    0.77   0.85   0.96]
    d0=[0.70, 0.18, 0.34, -0.38, -0.43, 0.38, -0.60, -0.12, 0.68, -0.030]

    axx = p0[1:d,1:d]\(diagm(d0[1:d])*p0[1:d,1:d])     # process model parameter
    vxx = [0.5^abs(i-j)*sqrt(0.8^(i+j)) for i=1:d,j=1:d]         # process variance
    sqvxx=real(sqrtm(vxx))
    axy = eye(d)     # observation model parameter
    vxy = sigma^2*eye(d)     # observation variance
    sqvxy=real(sqrtm(vxy))

    uvxx = reshape((I-kron(axx,axx))\vec(vxx),d,d)   # unconditional variance 
    uvxy = axy*uvxx*axy' + vxy
    squvxx = real(sqrtm(uvxx))
    squvxy = real(sqrtm(uvxy))

    NLG(axx,axy,sqvxx,sqvxy,squvxx,squvxy)
end

######################################################################################
### Implement HiddenMarkovModels's interface
### 
###     The user only needs to implement three methods:
###         - `rand_x` to simulate from x'|x  ("transition")
###         - `rand_y` to simulate from y'|x' ("measurement")
###         - `rand_ux` to simulate from a dominating measure on E_x (only necessary for low-rank methods)
######################################################################################


draw_x(model::NLG,x) = model.axx * x + model.sqvxx * randn(size(model.axy,1))
draw_y(model::NLG,x) = model.axy * x + model.sqvxy * randn(size(model.axy,2))


const scale=[1.2]

rand_ux(model::NLG) = scale[1]*model.squvxx*randn(size(model.axy,1))


######################################################################################
### Define methods
######################################################################################

function plot_filter(d_model,sigma,T,r,nx)

    model = nlg(d_model,sigma)

    dx,dy = size(model.axy)

    lg_model=HMM.LinearGaussianHMM(model.axx, model.sqvxx, model.axy, model.sqvxy)
    ini_data = (zeros(dx), zeros(dy))               # initial data for simulation
    ini_mean, ini_vcov = (zeros(dx), eye(dx))       # initial filter for Kalman filter
    ini_sample=[randn(dx) for _=1:100]              # initial filter for Particle and Kernel filters

    # preparing particle filter
    pf=PF(nx)

    # preparing kernel filter
    lr_kf = LRKKF(model,r,nx) 

    ### simulate data
    dxx,data_yy=rand(lg_model,ini_data,T) 

    ### Kalman filter
    kalman_filter_mean, kalman_filter_vcov =filtr(lg_model, (ini_mean, ini_vcov), data_yy)

    ### Kernel filter
    lr_kernel_filter = filtr(model,ini_sample,data_yy,lr_kf)
    # lr_kernel_filter_mean=[sum(lr_kernel_filter[j,t]*lr_kf.bxx[j] for j=1:nx) for t=1:T]
    # lr_kernel_filter_mean = HMM.mat(lr_kernel_filter_mean)

    ### Particle filter
    # particle_filter = filtr(lg_model,ini_sample,data_yy,pf)
    # particle_filter_mean = [mean(particle_filter[i,:,t]) for i=1:dx,t=1:T]

    # pl=plot(size=(1000,200*(dx)),layout=(dx,1),title="nx = $nx")
    # plot!(pl[1],1:T,kalman_filter_mean[1,:],l=(:black,2),label="Kalman filter")
    # plot!(pl[1],1:T,lr_kernel_filter_mean[1,:],l=(:orange,2),label="Kernel filter")
    # plot!(pl[1],1:T,particle_filter_mean[1,:],l=(:green,2),label="Particle filter")
    # for j=2:dx
    #     plot!(pl[j],1:T,kalman_filter_mean[j,:],l=(:black,2),label="")
    #     plot!(pl[j],1:T,lr_kernel_filter_mean[j,:],l=(:orange,2),label="")
    #     plot!(pl[j],1:T,particle_filter_mean[j,:],l=(:green,2),label="")
    # end
    # display(pl)

end


function monte_carlo(d_model,sigma,T,rr,nnx,n_mc=10)

    model = nlg(d_model,sigma)

    dx,dy = size(model.axy)

    lg_model=HMM.LinearGaussianHMM(model.axx, model.sqvxx, model.axy, model.sqvxy)
    ini_data = (zeros(dx), zeros(dy))               # initial data for simulation
    ini_mean, ini_vcov = (zeros(dx), eye(dx))       # initial filter for Kalman filter
    ini_sample=[randn(dx) for _=1:100]              # initial filter for Particle and Kernel filters


    n_nnx = length(nnx)
    n_rr = length(rr)
    mce_pf = zeros(n_mc,n_nnx)
    mct_pf = zeros(n_mc,n_nnx)
    mce_lr = zeros(n_mc,n_nnx,n_rr)
    mct_lr = zeros(n_mc,n_nnx,n_rr)

    for i=1:n_mc
        dxx,data_yy=rand(lg_model,ini_data,T) 
        kalman_filter_mean, kalman_filter_vcov =filtr(lg_model, (ini_mean, ini_vcov), data_yy)
        for i_nnx = 1:n_nnx
            tic()
            pf=PF(nnx[i_nnx])
            try
                particle_filter = filtr(lg_model,ini_sample,data_yy,pf)
                mct_pf[i,i_nnx] = toc()
                particle_filter_mean = [mean(particle_filter[i,:,t]) for i=1:dx,t=1:T]        
                mce_pf[i,i_nnx] = sqrt(mean((kalman_filter_mean - particle_filter_mean ).^2)) 
            catch
                mct_pf[i,i_nnx] = toc()    
                mce_pf[i,i_nnx] = 10_000. 
            end
            for i_rr = 1:n_rr
                tic()
                lr = LRKKF(model,rr[i_rr],nnx[i_nnx]) 
                lr_kernel_filter = filtr(model,ini_sample,data_yy,lr)
                mct_lr[i,i_nnx,i_rr] = toc()
                lr_kernel_filter_mean=[sum(lr_kernel_filter[j,t]*lr.bxx[j] for j=1:nnx[i_nnx]) for t=1:T]
                lr_kernel_filter_mean = HMM.mat(lr_kernel_filter_mean)
                mce_lr[i,i_nnx,i_rr] = sqrt(mean((kalman_filter_mean - lr_kernel_filter_mean ).^2)) 
            end
        end
    end

    mce_pf,mct_pf,mce_lr,mct_lr
end




######################################################################################
### Run
######################################################################################


d_model = 6
sigma   = 0.4
T  = 200
r  = 80 
nx = 1000 

# plot_filter(d_model,sigma,T,r,nx)


rr = [50, 150]
nnx = [500, 1000, 4000]

# mce_pf,mct_pf,mce_lr,mct_lr = monte_carlo(d_model,sigma,T,rr,nnx,1)

# @show squeeze(mean(mce_pf,1),1)
# @show squeeze(mean(mce_lr,1),1)



plot_filter(d_model,sigma,T,100,100_000)


# using ProfileView
# Profile.clear()
# @profile plot_filter(d_model,sigma,200,100,5000)
# ProfileView.view()




end #module

