struct DiscreteStrictHiddenMarkov <: StrictHiddenMarkovModel
    qxx::Matrix{Float64}
    qxy::Matrix{Float64}
end


draw_x(model::DiscreteStrictHiddenMarkov,x) = wsample(model.qxx[x,:])
draw_y(model::DiscreteStrictHiddenMarkov,x) = wsample(model.qxy[y,:])







#### "One-liner" filtering algorithm!

struct DiscreteHiddenMarkovModel
    a::Matrix{Float64}
    b::Matrix{Float64}
end

dhmm(a,b)=DiscreteHiddenMarkovModel(a,b)

function filtr(model::DiscreteHiddenMarkovModel,data,ini)
    T=length(data)
    nx=length(ini)
    fil=Array{Float64}(nx,T)
    fil[:,1]=ini
    for t=2:T
        fil[:,t]=upf(upq(fil[:,t-1],model.a),model.b[:,data[t]])
    end
    fil
end


