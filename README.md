# HiddenMarkovModels.jl

As of 10/2017, the package implements:
- A fairly complete set of methods for working with discrete hidden Markov models: filtering, smoothing, maximum likelihood estimation via closed-form EM (aka. the Baum-Welch algorithm), Viterbi smoothing. 
- Basic Kalman for linear Gaussian hidden Markov models: filtering, smoothing.
- Proof-of-concept nonlinear filtering via kernel filtering.

The tentative scope of the package is simulation / filtering / smoothing / parameter estimation / latent state inference for any type of hidden Markov models: discrete / continuous, linear / nonlinear, with or without feedback. So-called nonhomogeneous models (eg. Kalman filtering with time-dependent parameters) and control variables are out of scope. I don't have as much time as I would like for maintaining the package. If you may be interested in (paid) work on this package, please don't hesitate to get in touch.

## Overview


The package is organized around implicit interfaces: it defines abstract types such as `StrictHiddenMarkov` and implements methods such as `filtr(model::StrictHiddenMarkov,data,initial,technique::FilteringTechnique)`. The `filtr` method will in turn rely on the user's implementation of the suitable methods for the given type, such as one-step-ahead simulations or density evaluations.


| Models              | Filtering      | Smoothing | Viterbi           | Sampling | MLE | EM  |
| ------------------- | -------------- | --------- | ----------------- | -------- | --- | --- |
| Discrete            | X              | X         | X                 | X        | X   | X   |
| LinearGaussian      | X              | X         |                   |          |     |     |
| Strict              | KKF, KDF, PDF  |           |                   |          |     |     |
| AR                  | KKF, KDF, PDF  |           |                   |          |     |     |


## Usage

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



## Nitty-Gritty


Most filtering algorithms follow a recursive mutation-selection (or selection-mutation) strategy. This is simply because the true 


| Algorithm name          | mutation           | selection          | fixed basis? |
| ----------------------- | ------------------ | ------------------ | ------------ |
| KKF, kernel filtering   | kernel Markov rule | kernel Bayes rule  | yes          |
| KDF                     | kernel Markov rule | density evaluation | yes          |
| PKF                     | particles          | kernel Bayes rule  | no           |
| PDF, particle filtering | particles          | density evaluation | no           |


| Hidden Markov structure | order              | R=  | Q            | M          |
| ----------------------- | ------------------ | --- | ------------ | ---------- |
| General                 | selection-mutation | MQ  | Q(X'|X,y,y') | f(y'|X,y)  |
| Bootstrap               | mutation-selection | QM  | Q(X'|X,y)    | f(y'|X',y) |
| AR                      | mutation-selection | QM  | Q(X'|X)      | f(y'|X',y) |
| Strict                  | mutation-selection | QM  | Q(X'|X)      | f(y'|X')   |

<!-- 
| Models / Algorithms | Filtering  | Smoothing | Backward Sampling | Viterbi | MLE | EM  |
| ------------------- | ---------- | --------- | ----------------- | ------- | --- | --- |
| Discrete            | X          | X         | X                 | X       | X   | X   |
| LinearGaussian      | X          | X         |                   |         |     |     |
| Strict              | KF, BF, PF |           |                   |         |     |     |
| AR                  | KF, BF, PF |           |                   |         |     |     |
 -->
