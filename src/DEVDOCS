# Type Hierarchy

Concrete types are preceded by a (*)

## TYPES FOR MODELS

AbstractHiddenMarkovModel
	StrictHiddenMarkov
		*DiscreteStrictHiddenMarkov
		*LinearGaussian


## TYPES FOR FILTERING TECHNIQUES

FilteringTechnique
	KXF _(used for dispatching `initial_filter`)_
		*KKF_SHM
		*KDF_SHM
	*DiscreteFilter
	*KalmanFilter



# Relationship with other packages

`HiddenMarkovModels.jl` started as a thin wrapper on top of `DynamicDiscreteModels.jl`. Various probabilistic (filtering, smoothing, Viterbi) but also statistical (MLE and EM using `Optim.jl`) methods were implemented for the abstract type `DynamicDiscreteModel`. A user could implement a concrete type for her model of interest, inheriting from `DynamicDiscreteModel`, and immediately have access to those methods. A `HiddenMarkovModel` was such a concrete type, adding very little beyond `DynamicDiscreteModel`. But one could also implement concrete `DynamicDiscreteModel`'s with much more structure (in the sense of a nontrivial mapping from some "deep" statistical parameters to the transition matrix for the `DynamicDiscreteModel`, as commonly found in Economics).

In the spirit of making the package(s) more autonomous and easier to maintain, most of the code from `DynamicDiscreteModels.jl` was subsequently brought in `HiddenMarkovModels.jl` and `DynamicDiscreteModels.jl` was deprecated. Dependencies on optimization packages were dropped: the scope of the package may now include likelihood evaluation, but not likelihood _optimization_. 

`DynamicDiscreteModel` used to inherit from `StatsBase.StatisticalModel`, ie. originally we had:

`HiddenMarkovModels.HiddenMarkovModel <: DynamicDiscreteModels.DynamicDiscreteModel <: StatsBase.StatisticalModel`

It is not the case anymore. The `StatsBase.StatisticalModel` interface is defined [here](https://github.com/JuliaStats/StatsBase.jl/blob/master/docs/src/statmodels.md). There are at least two reasons for not inheriting:

(1) `HiddenMarkovModel`'s are designed as _probabilistic_ models rather than _statistical_ models. The data generating process is fixed. There is no reasoning about a parametrization from some statistical parameter θ to the DGP. To use a `HiddenMarkovModel` in a statistical context, I recommend a design along the following lines:

~~~
MyStatModel{T<:HiddenMarkovModel}
	core::T
	parameter::Vector{Float64}
end


function coef!(model::MyStatModel,parameter)
	# modify `model.core`
end

function rand(model::MyStatModel,T)
	rand(model.core,T)
end

function loglikelihood(model::MyStatModel)
	loglikelihood(model.core)
end

~~~  

(2) Some of the methods contracted by the `StatsBase.StatisticalModel`'s interface are not applicable (`adjr2`?). The list of methods (09/2017):

~~~
adjr2, aic, aicc, bic, coef, coeftable, confint, deviance, dof, fit, fit!, loglikelihood, nobs, nulldeviance, r2, stderr, vcov
~~~

Note that `StatsBase` is now the owner of `loglikelihood` (I think it used to be `Distributions.jl`).

