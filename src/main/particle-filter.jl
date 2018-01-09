immutable BootstrapParticleFilter  <: FilteringTechnique
    n::Int   #number of particles
end

PF(n)=BootstrapParticleFilter(n)


doc"""
    filtr(transition,emission,initial,data,filteringalgo::BootstrapParticle)

Run a bootstrap particle filter in a model with strict hidden Markov dynamics.
The method runs an optimal particle filter in the artificial model where measurement is delayed one period.
As a result the optimal particle filter tracks ``x_{t+1}|y_{1:t}``. 
The method does a last step adjustement to account for the observation ``y_T``.
`transition(x)` must return a random draw from ``Q(x'|x)``. 
`emission(x,y)` must return the density of ``Q(y|x)`` with respect to some fixed dominating measure ``\lambda(dy)``.
`initial(y)` must return a random draw from the conditional initial distribution ``x_1|y_1``.
"""
function filtr(model,ini_sample,data,pf::BootstrapParticleFilter)
    n=pf.n
    nx=length(ini_sample[1])
    T=length(data)
    xx=zeros(nx,n,T)   #xx[i,j,t] is the ith coordinate of the jth particle at time t
    initial_index=sample(1:length(ini_sample),n)
    for i=1:n
        xx[:,i,1]=ini_sample[initial_index[i]]
    end
    xxx=copy(xx[:,:,1])
    w=zeros(n)
    for t=2:T
        for i=1:n
            xxx[:,i]=draw_x(model,xx[:,i,t-1])
            w[i]=cpdf(model,Val(:y),xxx[:,i],data[t])
        end

        ## resampling step
        # normalize!(w,1) #I believe this may unnecessary as wsample can take unnormalized weights
        indx=wsample(1:n,w,n)
        xx[:,:,t]=xxx[:,indx]
    end
    xx
end


# struct ParticleFilter  <: FilteringTechnique
#     n::Int   #number of particles
# end

# doc"""
#     filter(transition,emission,initial,data,filteringalgo::Particle)

# Run a particle filter in a model with general hidden Markov dynamics (partially observed Markov chain).
# `transition(x,y,y')` must return a random draw from ``Q(x'|x,y,y')``. 
# `emission(x,y,y')` must return the density of ``Q(y'|x,y)`` with respect to some dominating measure ``\lambda(y'|y)``.
# `initial(y)` must return a random draw from the conditional initial distribution ``x_1|y_1``.
# """
# function filtr(transition,emission,initial,data,filteringalgo::ParticleFilter)
#     n=filteringalgo.n
#     T=length(data)
#     xx=zeros(n,T)
#     xx[:,1]=[initial(data[1]) for _=1:n]
#     xxx=copy(xx[:,1])
#     w=zeros(n)
#     for t=2:T
#         map!(x->emission(x,data[t-1],data[t]),w,view(xx,:,t-1))
#         normalize!(w,1) #I believe this may unnecessary as wsample can take unnormalized weights
#         Distributions.wsample!(view(xx,:,t-1),w,xxx)
#         map!(x->transition(x,data[t-1],data[t]),view(xx,:,t),xxx)
#     end
#     xx
# end
