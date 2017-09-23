abstract type AbstractHiddenMarkovModel end


######################################################################################
### Implicit interface that all AbstractHiddenMarkovModels must implement
######################################################################################

## FIELDS

# # (no fields)

## METHODS

# # Sample one step ahead starting at today's z=(x,y) 
# rand{S1,S2}(model::StrictHiddenMarkovModel,xy::Tuple{S1,S2})



######################################################################################
### Methods common to all AbstractHiddenMarkovModel
######################################################################################

function rand{Tx,Ty}(model::AbstractHiddenMarkovModel,initial::Tuple{Tx,Ty},T::Int)
    xx=Vector{Tx}(T)
    yy=Vector{Ty}(T)
    xx[1]=initial[1]
    yy[1]=initial[2]
    for t=1:T-1
        xx[t+1],yy[t+1]=rand(model,(xx[t],yy[t]))
    end
    xx,yy
end


######################################################################################
### Types and Functions that will be used for defining subtypes 
######################################################################################



abstract type FilteringTechnique
end

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



# THIS NEEDS TO BE CLEANED UP: no all AbstractHiddenMarkovModels call rand(model,flag,x)
# This works only for continuous models where each observation is different (does it?)
# to do: make the kernel `kk` an option, maybe by adding `model` as an argument
function markovapprox(model,flag,bxx,byy,ky,m=1000,tol=0.1)
    nx=length(bxx)
    ny=length(byy)
    q=zeros(nx,ny)
    for ix=1:nx
        print("$ix ")
        gi=zeros(ny)
        for j=1:m
            y=rand(model,flag,bxx[ix])
            for jy=1:ny
                gi[jy]=gi[jy]+kk(byy[jy],y)/m
            end            
        end
        q[ix,:]=probnorm((ky+tol*I/sqrt(ny))\gi)
    end
    q
end

rand(model::AbstractHiddenMarkovModel,flag::Val{:x},x) = draw_x(model,x)
rand(model::AbstractHiddenMarkovModel,flag::Val{:y},x) = draw_y(model,x)