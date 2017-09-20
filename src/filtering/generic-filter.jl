######################################################################################
### So far the generic filter here is common to all algorithms working "in a basis".
### This includes discrete filtering, basis filtering, and kernel filtering.
### With some more work it can probably be made to work for particle filtering and Kalman filtering.
######################################################################################



# - Initial nonlinear filter is given as a sample, must be expressed in basis
# - approximation of the dynamics have been computed
function filtr(model,data,ini_sample,filtering_method)
    ini = initial_filter(model,ini_sample,filtering_method)
    _filtr(model,data,ini,filtering_method)
end



function _filtr(model,data,ini_filter,filtering_method)
    T=length(data)
    nx,ny=size(filtering_method.qxy)

    fil=Array{Float64}(nx,T)
    fil[:,1]=ini_filter
    predictive=zeros(nx)

    print("Running the generic filter... ")
    for t=1:T-1
        print((t%10==0)?"$t ":"")
        
        fil[:,t+1] = filter_update(model,fil[:,t],data[t],data[t+1],filtering_method)
        
    end
    println()

    fil
end

