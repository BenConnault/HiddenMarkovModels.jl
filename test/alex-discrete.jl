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

# check forward-backward algorithm on perfect sequence
alpha,beta,log_p_obs = forward_backward(model,o; scaling=false)
@test(log_p_obs == 0.0)
@test(all(alpha[:,1] .== 1))
@test(all(alpha[:,2] .== 0))
@test(all(beta[end,:] .== 1))
@test(all(beta[:,1] .== 1))

# test forward-backward with scaling
alpha,beta,log_p_obs = forward_backward(model,o)
@test(log_p_obs == 0.0)
@test(all(alpha[:,1] .== 1))
@test(all(alpha[:,2] .== 0))
@test(all(beta[end,:] .== 1))
@test(all(beta[:,1] .== 1))

# check viterbi algorithm on perfect sequence
@test(all(viterbi(model,o) .== s))

# check forward-backward algorithm on impossible sequence
alpha,beta,log_p_obs = forward_backward(model,o+1; scaling=false)
@test(log_p_obs == -Inf)
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

alpha,beta,log_p_obs = forward_backward(model, o; scaling=false)

@test(all(round(alpha,6) .== round(true_alpha',6)))
@test(all(round(beta,5) .== round(true_beta',5)))

# check forward/backward (with scaling)
alpha_scaled = [ 0.625   0.7332  0.8173  0.5884  0.3212 ;
                 0.375   0.2668  0.1827  0.4116  0.6788 ]

beta_scaled = [  4.5729   3.5946   2.9391   2.4312  3.2847 ;
                 12.3785  16.6108  23.6922  7.1028  3.2847 ]

alpha,beta,log_p_obs = forward_backward(model,o; scaling=true)

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

