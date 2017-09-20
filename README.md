# HiddenMarkovModels.jl

As of 10/2017, the package implements:
- A fairly complete set of methods for working with discrete hidden Markov models.
- Basic Kalman filtering.
- Experimental approximate nonlinear filtering algorithms via kernel filtering.
The tentative scope of the package is simulation / nonlinear filtering / parameter estimation / latent state inference for hidden Markov models. I don't have as much time as I would like for growing the package. If you may be interested in (paid) work on this package, please don't hesitate to get in touch.

##Overview


The package is organized around implicit interfaces: it defines abstract types such as `StrictHiddenMarkov` and implements methods such as `filtr(model::StrictHiddenMarkov,data,initial,technique::FilteringTechnique)`, relying on the user's implementation of the suitable methods for the given type, such as `randa(model::StrictHiddenMarkov,x)` to draw next period's random unobserved state x_t+1 given today's unobserved state x_t=x, and similarly for drawing an observation. 

| Models / Algorithms | Filtering  | Smoothing | Backward Sampling | Viterbi | MLE | EM  |
| ------------------- | ---------- | --------- | ----------------- | ------- | --- | --- |
| Discrete            | X          | X         | X                 | X       | X   | X   |
| LinearGaussian      | X          | X         |                   |         |     |     |
| Strict              | KF, BF, PF |           |                   |         |     |     |
| AR                  | KF, BF, PF |           |                   |         |     |     |


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
