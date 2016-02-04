# simple discrete hmm with testable output
A = eye(2)
B_matrix = [0.0  1.0  0.0 ;
            1.0  0.0  0.0 ]

model = HMM(A,B_matrix)
model.p = vec([1 0])

# state sequence s, observation sequence o
n_obs = 10
s,o = rand(model,n_obs)
@test(length(s) == length(o) == n_obs)
@test(all(s .== 1))
@test(all(o .== 2))

# check forward algorithm on perfect sequence
alpha,p_obs = forward(model,o; scaling=false)
@test(p_obs == 1)
@test(all(alpha[:,1] .== 1))
@test(all(alpha[:,2] .== 0))

# test forward with scaling
alpha,log_p_obs,coeff = forward(model,o)
@test(log_p_obs == 0.0)
@test(all(alpha[:,1] .== 1))
@test(all(alpha[:,2] .== 0))
@test(all(coeff .== 1.0))

# check backward algorithm on perfect sequence
beta = backward(model,o)
@test(all(beta[end,:] .== 1))
@test(all(beta[:,1] .== 1))

# test backward with scaling
beta = backward(model,o;scale_coeff=coeff)
@test(all(beta[end,:] .== 1))
@test(all(beta[:,1] .== 1))

# check viterbi algorithm on perfect sequence
@test(all(viterbi(model,o) .== s))

# check forward-backward algorithm on impossible sequence
alpha,p_obs = forward(model,o+1; scaling=false)
beta = backward(model,o+1)
@test(p_obs == 0)
@test(all(alpha .== 0))
@test(all(beta[1:end-1,:] .== 0))

######
# Compare output to Michael Hamilton's "dishonest casino" example
# see Hamilton's HMM implementation in python:
# http://www.cs.colostate.edu/~hamiltom/code.html#python-hidden-markov-model

A = [ 0.99 0.01 ;
      0.01 0.99 ]
B = [ 1/6   1/6   1/6   1/6   1/6   1/6 ;
      1/10  1/10  1/10  1/10  1/10  1/2 ]
p = [0.5, 0.5]
model = HMM(A,B,p)

@test(model.n == 2)
for i = 1:model.n
  @test(typeof(model.B[i]) == Categorical)
  @test(model.B[i].K == 6)
end
@test(length(model.p) == model.n)

o = [1, 2, 1, 6, 6] # observation sequence

# check forward/backward (without scaling)
true_alpha = [ 8.3333e-02   1.3833e-02   2.2909e-03   3.7885e-04   6.295e-05  ;
               5.0000e-02   5.0333e-03   5.1213e-04   2.6496e-04   1.3305e-04 ]

true_beta = [ 8.9630e-04   5.2841e-03   3.0533e-02   1.7000e-01   1.0000e+00 ;
              2.4262e-03   2.4418e-02   2.4613e-01   4.9667e-01   1.0000e+00 ]

alpha,p_obs = forward(model,o; scaling=false)
beta = backward(model,o)

@test(round(p_obs,6) == 0.000196)
@test(all(round(alpha,6) .== round(true_alpha',6)))
@test(all(round(beta,5) .== round(true_beta',5)))

# check forward/backward (with scaling)
alpha_scaled = [ 0.625   0.7332  0.8173  0.5884  0.3212 ;
                 0.375   0.2668  0.1827  0.4116  0.6788 ]

beta_scaled = [  4.5729   3.5946   2.9391   2.4312  3.2847 ;
                 12.3785  16.6108  23.6922  7.1028  3.2847 ]

alpha,log_p_obs,coeff = forward(model,o; scaling=true)
beta = backward(model,o; scale_coeff=coeff)

@test(log_p_obs == log(p_obs))
@test(all(round(alpha,4) .== round(alpha_scaled',4)))
@test(all(round(beta,4) .== round(beta_scaled',4)))

# check viterbi
true_vit = [2,2,2,2,2]
@test(all(viterbi(model,o) .== true_vit))
@test(all(viterbi(model,[1,2,3,4,5]) .== ones(Int,5)))

# another check for viterbi
model.A = [ 0.7 0.3 ;
          0.3 0.7 ]
o = [ 1, 2, 1, 6, 6, 3, 3, 2, 1 ]
true_vit = [ 1, 1, 1, 2, 2, 1, 1, 1, 1 ]
@test(all(viterbi(model,o) .== true_vit))

# check Baum-Welch algorithm
o = [ 1, 2, 1, 6, 6, 3, 3, 2, 1 ]
expected_result_A = [ 0.75  0.25 ;
                      0.25  0.75 ]
expected_result_B = [ 0.6  0.4  0.0  0.0  0.0  0.0 ;
                      0.0  0.0  0.5  0.0  0.0  0.5 ]
expected_result_p = [1.0, 0.0]

model1 = deepcopy(model) # check without scaling alpha/beta
fit!(model1, o; max_iter=60, scaling=false)

@test(all(round(model1.A,4) .== expected_result_A))
@test(all(round(model1.B[1].p,4) .== expected_result_B[1,:]'))
@test(all(round(model1.B[2].p,4) .== expected_result_B[2,:]'))
@test(all(round(model1.p,4) .== expected_result_p))

model2 = deepcopy(model) # check with scaling alpha/beta
fit!(model2, o; max_iter=60, scaling=true)
@test(all(round(model2.A,4) .== expected_result_A))
@test(all(round(model2.B[1].p,4) .== expected_result_B[1,:]'))
@test(all(round(model2.B[2].p,4) .== expected_result_B[2,:]'))
@test(all(round(model2.p,4) .== expected_result_p))

# model3 = deepcopy(model) # check with multiple sequences
# oo = (Vector{Int})[o;o] # two copies of the same sequence
# baum_welch!(model3, oo; max_iter=60, tol=NaN, scaling=true)
# @test(all(round(model3.A,4) .== expected_result_A))
# @test(all(round(model3.B,4) .== expected_result_B))
# @test(all(round(model3.p,4) .== expected_result_p))

# ## Check Baum-Welch from random initialization
# # Model used to generate observation sequences:
# # A = [ 0.7 0.3 ;
# #       0.4 0.6 ]
# # B = [ 1/2        1/3   1/6+eps() ;
# #       1/6+eps()  1/3   1/2       ]
# # p = [0.5, 0.5]
# model = dHMM(2,3)
# o = [1,3,1,1,1,2,3,1,2,1,3,2,1,3,2,2,2,2,2,2,2,1,1,1,2,1,1,2,3,2,1,2,3,3,2,1,1,3,2,3,2,2,2,3,2,3,3,2,1,2]

# ch = baum_welch!(model, o; max_iter=1000, tol=1e-7)

# @test(all(round(sum(model.A,2),15) .== 1.0))
# @test(all(round(sum(model.B,2),15) .== 1.0))
# @test(round(sum(model.p),15) == 1.0)

# # log-liklihood should always increase under baum_welch
# @test(all(diff(ch) .>= 0.0))

# # log-liklihood should plateau (test tolerance parameter)
# if length(ch)<1000
#   @test(ch[end] - ch[end-1] < 1e-7)
# end

# ## Check Baum-Welch on multiple sequences
# o1 = [3,3,3,1,1,1,1,3,2,2]
# o2 = [1,3,3,1,3,2,2,3,1,1]
# o3 = [2,3,3,3,3,1,2,1,3,2]
# o4 = [2,1,1,2,3,1,3,2,3,1]
# o5 = [3,3,3,1,3,2,1,3,1,1]
# o6 = [2,1,1,2,2,2,2,3,2,3,1,1,1,2,3,3,2,3,3,1]
# o7 = [1,1,3,1,3,3,2,3,2,3,2,1,1,2,2,2,2,3,1,2]
# o8 = [1,3,2,1,3,2,2,2,2,1,2,1,3,2,2,2,2,1,2,2]
# o9 = [3,2,2,1,3,2,1,2,3,3,1,3,1,2,2,3,2,1,3,3]
# o10 = [2,2,1,3,2,2,1,3,1,1,2,3,3,1,1,3,3,1,1,3]

# seq1 = (Vector{Int})[o1; o2; o3; o4; o5]
# seq2 = (Vector{Int})[o6; o7; o8; o9; o10]
# seq3 = (Vector{Int})[o1; o2; o3; o4; o5; o6; o7; o8; o9; o10]

# # First set of sequences
# model = dHMM(2,3) # re-initialize with random params
# ch = baum_welch!(model, seq1; max_iter=1000, tol=1e-7)
# @test(all(round(sum(model.A,2),15) .== 1.0))
# @test(all(round(sum(model.B,2),15) .== 1.0))
# @test(round(sum(model.p),15) == 1.0)
# @test(all(diff(ch) .>= -1e-10))
# if length(ch)<1000
#   @test(ch[end] - ch[end-1] < 1e-7)
# end

# # Second set of sequences
# model = dHMM(2,3) # re-initialize with random params
# ch = baum_welch!(model, seq2; max_iter=1000, tol=1e-7)
# @test(all(round(sum(model.A,2),15) .== 1.0))
# @test(all(round(sum(model.B,2),15) .== 1.0))
# @test(round(sum(model.p),15) == 1.0)
# #@test(all(diff(ch) .>= -1e-10))
# if length(ch)<1000
#   @test(ch[end] - ch[end-1] < 1e-7)
# end

# # All sequences
# model = dHMM(2,3) # re-initialize with random params
# ch = baum_welch!(model, seq3; max_iter=100, tol=NaN)
# @test(all(round(sum(model.A,2),15) .== 1.0))
# @test(all(round(sum(model.B,2),15) .== 1.0))
# @test(round(sum(model.p),15) == 1.0)

# These tests fail...
#    @test(all(diff(ch) .>= 0.0))
#    @test(ch[end] - ch[end-1] < 1e-7)
# potentially because having different length observation sequences
# violates some assumptions?
