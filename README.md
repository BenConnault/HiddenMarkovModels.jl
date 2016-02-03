# HiddenMarkovModels.jl

Basic simulation / parameter estimation / latent state inference for hidden Markov models. Thin front-end package built on top of [DynamicDiscreteModels.jl](https://github.com/BenConnault/DynamicDiscreteModels.jl).

## Installation

~~~julia
julia> Pkg.add("HiddenMarkovModels")
~~~

##Usage

Let's take a hidden Markov model with hidden/latent state x and emission/observed sate y. x can takes values 1,2, and y can take values 1, 2, 3. x has transition matrix A and y is drawn conditional on x according to the matrix B below:

~~~
.
    [ .4   .6 ]        [ .3   .1   .6 ]
A = [ .3   .7 ]    B = [ .5   .2   .3 ]
~~~    

Here is the Julia code to initiate the model, draw 10,000 consecutive observations from it, and estimate the value of A and B from the data using the Baum-Welch algorithm. The Baum-Welch algorithm is just the standard EM algorithm, where the optimization step (the M step) is accessible in closed-form thanks to the particular structure of hidden Markov models. We obtain the maximum likelihood estimator.   


~~~julia
julia> using HiddenMarkovModels;
julia> a=[.4 .6; .3 .7];
julia> b=[.3 .1 .6; .5 .2 .3];
julia> model=hmm((a,b));
julia> data=rand(model,10000);
julia> @time abhat=em(model,data)
estimating hidden Markov model via Baum-Welch algorithm...
 log-likelihood: -1.0267
 0.004404 seconds (102 allocations: 317.891 KB)
(
2x2 Array{Float64,2}:
0.398849  0.601151
0.300235  0.699765,

2x3 Array{Float64,2}:
0.301938  0.099349  0.598713
0.501222  0.199013  0.299765)
~~~

In this example it took 4.4 ms to compute the MLE from 10,000 time series observations. The heavy-lifting is done in the back-end package  [DynamicDiscreteModels.jl](https://github.com/BenConnault/DynamicDiscreteModels.jl).
