######################################################################################
### Implementation of the HiddenMarkovModel interface
######################################################################################

abstract type DiscreteHMM <: HiddenMarkovModel
end


######################################################################################
### Implicit interface that all DiscreteHMM's must implement
######################################################################################

## FIELDS

    # nx::Int
    # ny::Int

## METHODS

## (i) passed on from AbstractHiddenMarkovModel:

### Sample one step ahead starting at today's z=(x,y) 
# rand{S1,S2}(model::StrictHiddenMarkovModel,xy::Tuple{S1,S2})

## (ii) new methods:

### evaluate the transition matrix Q[x,y,x',y']=Q(x',y'|x,y)
qxyxy(model::DiscreteHMM,x,y,x2,y2) = error("no method qxyxy!(model::$(typeof(model)),x,y,x2,y2)")


######################################################################################
### Methods common to all DiscreteHMM's
######################################################################################


function rand(model::DiscreteHMM,xy::Tuple)
    x,y = xy
    w = vec([qxyxy(model,x,y,jx,jy) for jx=1:model.nx, jy=1:model.ny])
    xy2 = ind2sub((model.nx,model.ny),wsample(w)) 
    xy2
end

#Should I try and make this type-stable?
# => function _filtr{F}(model::DiscreteHMM{F},ini_filter,data)

function _filtr(model::DiscreteHMM,ini_filter,data)
    T=length(data)
    nx = length(ini_filter)

    el_type = eltype(qxyxy(model,1,1,1,1))

    fil=Array{el_type}(nx,T)   #Does it need to be Float64?
    fil[:,1]=ini_filter
    llk = 0.

    print("Running discrete filter... ")
    for t=1:T-1
        rho=0.0
        for jx=1:nx
            fil[jx,t+1]=0.
            for ix=1:nx
                fil[jx,t+1] += qxyxy(model,ix,data[t],jx,data[t+1])*fil[ix,t]
            end
            rho += fil[jx,t+1]
        end
        for jx=1:nx
            fil[jx,t+1] /= rho
        end
        llk += log(rho)
    end
    println()
    fil,llk/T
end

filtr(model::DiscreteHMM,ini_filter,data)=_filtr(model,ini_filter,data)[1]

"""
    loglikelihood(model::DiscreteHMM,ini_filter,data)

Return the conditional log-likelihood ``log p(y_{2:T}|y_1)``. 
`ini_filter` contains the initial filter ``p(x_1|y_1)``.
"""
loglikelihood(model::DiscreteHMM,ini_filter,data)=_filtr(model,ini_filter,data)[2]


function _smoother(model::DiscreteHMM,fil,data)
    nx,T = size(fil)

    el_type = eltype(qxyxy(model,1,1,1,1))


    smo = Array{Float64}(nx,T)  # stores p(X_t | y_{1:T})
    joi = Array{Float64}(nx,nx,T-1)  # joi[:,:,t] stores p(X_t X_t+1 | y_{1:T})  
    # storing a (nx x nx) matrix is not strictly necessary: 
    #    - we could loop over the columns of `cod`, ie over x_{t+1} = jx for more efficient pure smoothing
    #    - however here we also get the eweights


    smo[:,T]=view(fil,:,T)

    print("Running discrete smoother... ")
    for t=T-1:-1:1
        for jx=1:nx
            rho = 0.0
            for ix=1:nx
                joi[ix,jx,t] = qxyxy(model,ix,data[t],jx,data[t+1])*fil[ix,t]
                rho += joi[ix,jx,t]
            end
            w = smo[jx,t+1]/rho
            for ix=1:nx
                joi[ix,jx,t] *= w
                smo[ix,t]    += joi[ix,jx,t]
            end
        end
    end
    println()
    smo, joi
end


function filter_smoother(model::DiscreteHMM,ini_filter,data)
    fil, llk = _filtr(model,ini_filter,data)
    smo, joi = _smoother(model,fil,data)
    fil,smo
end



# function eweights!(w,model::DiscreteHMM,ini_filter,data)
#     fil, llk = _filtr(model,ini_filter,data)
#     smo, joi = _smoother(model,fil,data)
#     nx,T = size(fil)

#     joi
# end

