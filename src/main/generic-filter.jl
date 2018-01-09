######################################################################################
### So far the generic filter here is common to all algorithms working "in a basis".
### This includes kernel-kernel filters and kernel-density filters
### (as well as discrete filters, although those have an alternative specialized routine).
### With some more work it can probably be made to work for particle filtering and Kalman filtering.
######################################################################################



# Any method that uses Kernel method for mutation, 
# so far: KKFs and KDFs.
# Don't want to use a union type because these are KKF_GHM, KKF_BHM, KKF_AHM, KKF_SHM,  
abstract type KXF <: FilteringTechnique
end

function initial_filter(model,ini_sample,filtering_method::KXF)
    n=length(ini_sample)
    g=gramian(filtering_method.bxx,ini_sample)*ones(n)/n
    probnorm(filtering_method.kx\g)
end

abstract type LRKXF <: FilteringTechnique
end


function initial_filter(model,ini_sample,kf::LRKXF)
    n=length(ini_sample)
    v6 = ones(n)/n
    w = kf.gx20'*kf.gx20
    gx60 = gramian(ini_sample,kf.xx0)
    probnorm(kf.gx20*(w\(gx60'*v6)))
end


# Initial nonlinear filter is typically given as a sample, must be expressed in basis
function filtr(model,ini_sample,data,filtering_method)
    ini = initial_filter(model,ini_sample,filtering_method)
    _filtr(model,ini,data,filtering_method)
end



function _filtr(model,ini_filter,data,filtering_method)
    T=length(data)
    nx = length(ini_filter)

    fil=Array{Float64}(nx,T)
    fil[:,1]=ini_filter

    pred = zeros(nx) 

    print("Running generic filter... ")
    for t=1:T-1
        # print((t%10==0)?"$t ":"")
        # println("$t ")
        
        filter_update!(view(fil,:,t+1),pred,model,view(fil,:,t),data[t],data[t+1],filtering_method)
        
    end
    println()

    fil
end

