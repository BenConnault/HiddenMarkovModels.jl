NOW
- Kalman filter
- Tests
- switch ini and data in filtr call

LATER
- Optimize `rkhs.jl/kbr`. This is not super important because performance critical applications will use the low-rank version of `kbr` anyway.
- Port all the code for discrete models from the old `DynamicDiscreteModels.jl` to the new thing. This includes: smoothing, EM, Baum-Welch, Viterbi.


EVEN LATER
- Design `generic-filter.jl/filtr` to allow for other filtering techniques such as particles and Kalman. This will probably require implementing a `Filter` type, since right now it's simply a vector of weights.