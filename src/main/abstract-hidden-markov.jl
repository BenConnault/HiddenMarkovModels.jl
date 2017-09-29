abstract type HiddenMarkovModel end


######################################################################################
### Implicit interface that all HiddenMarkovModels must implement
######################################################################################

## FIELDS

# # (no fields)

## METHODS

# # Sample one step ahead starting at today's z=(x,y) 
# rand{S1,S2}(model::StrictHiddenMarkovModel,xy::Tuple{S1,S2})



######################################################################################
### Methods common to all HiddenMarkovModel
######################################################################################

function rand{Tx,Ty}(model::HiddenMarkovModel,initial::Tuple{Tx,Ty},T::Int)
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

