# Type Hierarchy

Concrete types are preceded by a (*).

Types for models:

~~~
HiddenMarkovModel
	DiscreteHMM
	StrictHMM
		*LinearGaussianHMM
DynamicDiscreteModel
	*DHMM
~~~


Types for filtering techniques:

~~~
FilteringTechnique
	KXF
		*KKF_SHMM
		*KDF_SHMM
~~~

`KXF` is a parent for any filtering technique that represents the filter in a RKHS basis. It is used to dispatch initialization of the filter by passing an initial sample.  

# TODO

NOW
- add tests for utils
- particle filtering
- KDF filtering
- High-dim stuff

LATER
- Discrete `x` / continuous `y` closed form filtering.
- Port Viterbi and `eweights!()` filtering from DDM to new back-end. Delete DDM.
- Kalman likelihood.
- singular variance Kalman


EVEN LATER
- Port explicit Jacobian code from `DynamicDiscreteModels.jl`. Not clear anybody would need this now that the code supports automatic differentiation. 
- add various methods for working with Markov chains: isreversible(), stationary(), etc.
- Design `generic-filter.jl/filtr` to allow for other filtering techniques such as particles and Kalman. This will probably require implementing a `Filter` type, since right now it's simply a vector of weights.

# Relationship with other packages

`HiddenMarkovModels.jl` started as two packages:
- `DynamicDiscreteModels.jl` implemented various probabilistic (filtering, smoothing, Viterbi) but also statistical (MLE and EM using `Optim.jl`) methods for the abstract type `DynamicDiscreteModel`.
- `DynamicDiscreteModel` inherited from `StatsBase.StatisticalModel`.
- `DynamicDiscreteModel` was designed to be compatible with arbitrary parametrizations of the transition matrices in terms of "deep statistical parameters", as commonly found in Economics.
- `HiddenMarkovModels.jl` was a thin wrapper on top of `DynamicDiscreteModels.jl`, implementing a simple `HiddenMarkovModel` type that was carrying a transition matrix and a measurement matrix -- the simplest parametrization one can imagine.

In the spirit of making the package(s) more autonomous and easier to maintain:
- code from `DynamicDiscreteModels.jl` was subsequently brought in `HiddenMarkovModels.jl` and `DynamicDiscreteModels.jl` was deprecated.
- dependencies on optimization packages were dropped: the scope of the package may now include likelihood evaluation, but not likelihood _optimization_. 
- types do not inherit from `StatsBase.StatisticalModel` anymore. 


The `StatsBase.StatisticalModel` interface is defined [here](https://github.com/JuliaStats/StatsBase.jl/blob/master/docs/src/statmodels.md).  The list of methods (09/2017):

~~~
adjr2, aic, aicc, bic, deviance, dof, nulldeviance, r2, 
stderr, vcov, fit, fit!, loglikelihood, nobs, coef, coeftable, confint
~~~

Note that `StatsBase` is now the owner of `loglikelihood` (I believe it used to be `Distributions.jl`).

