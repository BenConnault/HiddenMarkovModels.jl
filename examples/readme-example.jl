# 09/2017 
# Nonlinear filtering for censored AR(1)


module dev

using HiddenMarkovModels
HMM=HiddenMarkovModels



######################################################################################
### Model definition
###   
### - the unobserved variable x_t is an AR(1) with persistence parameter rho 
### - the observed variable y_t = censor(x_t + noise)
### 
######################################################################################


rho = 0.999

struct TAR <: HMM.StrictHMM
end

censor(x) =  min(max(-1,x),1)

# Implement the `HMM.StrictHMM` interface:
HMM.draw_x(m::TAR,x) = [rho*x[1]+0.2*randn()]
HMM.draw_y(m::TAR,x) = [censor(x[1]+0.2*rand())]


# create model
tar_model=TAR()

# Choose initial values and simulate. 
x0  = 0.0
y0  = 0.0
ini = ([x0],[y0])

T     = 500
xx,yy = rand(tar_model,ini,T)

# Pick an approximate nonlinear filtering technique - here kernel filtering on a simple grid:
bxx = [[x] for x=linspace(-15,15,100)]
byy = [[y] for y=linspace(-15,15,100)]
kf  = KKF(tar_model,bxx,byy,1000)

# pass an initial value for the nonlinear filter p(x_1|y_1) in the form of a sample.
# Here we initialize the nonlinear filter at p(x_1|y_1) = delta_{x_0} (a point mass).
#   so we pass the "sample" with one observation [ [x_0] ]:
ini = [ [x0] ]


# run the nonlinear filter:
nl_filter=filtr(tar_model,ini,yy,kf)


# We are done, now we just need to plot it.

using StatsBase.Weights
using Plots
plotly()

#q^th quantiles of the i^th coordinate
function nlf2quantile(nlf,bxx,q,i=1)
	T=size(nl_filter,2)
	qq=zeros(T)
	xx=[x[i] for x=bxx]
	for t=1:T
		qq[t]=quantile(xx,Weights(nlf[:,t]),q)
	end
	qq
end

devec(yy,i=1)=[y[i] for y=yy]


qq1 = nlf2quantile(nl_filter,bxx,0.025)
qq2 = nlf2quantile(nl_filter,bxx,0.5)
qq3 = nlf2quantile(nl_filter,bxx,0.975)


# qq21 = nlf2quantile(nl_filter2,bxx,0.025)
# qq22 = nlf2quantile(nl_filter2,bxx,0.5)
# qq23 = nlf2quantile(nl_filter2,bxx,0.975)


pl=plot(layout=(2,1),size=(1000,400))
plot!(pl[1,1],1:T,devec(yy),l=(:grey),lab="data")
plot!(pl[2,1],1:T,qq1,l=(:blue,0.5),lab="")
plot!(pl[2,1],1:T,qq2,l=(:blue),lab="filter with 95% band")
plot!(pl[2,1],1:T,qq3,l=(:blue,0.5),lab="")
plot!(pl[2,1],1:T,devec(xx),l=(:black),lab="true unobserved state")
# plot!(pl[3,1],1:T,qq21,l=(:blue,0.5),lab="")
# plot!(pl[3,1],1:T,qq22,l=(:blue),lab="filter with 95% band")
# plot!(pl[3,1],1:T,qq23,l=(:blue,0.5),lab="")
display(pl)







end