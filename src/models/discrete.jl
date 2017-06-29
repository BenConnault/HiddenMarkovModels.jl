struct DiscreteHiddenMarkovModel
    a::Matrix{Float64}
    b::Matrix{Float64}
end


dhmm(a,b)=DiscreteHiddenMarkovModel(a,b)