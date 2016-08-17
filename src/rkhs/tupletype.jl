typealias NT{T1,T2} Union{T1,Tuple{Vararg{Union{T1,T2}}}}
typealias NestedTuple{AtomicType} NT{AtomicType,NT{AtomicType,NT{AtomicType,AtomicType}}}

#Any Type <: NestedTuple! But this is a scope reminder for the human reader.  
function instantiate{H <: NestedTuple}(::Type{H})
	H()
end

function instantiate{H <: Tuple{Vararg{NestedTuple}}}(::Type{H})
	if length(H.parameters) > 1
		return (instantiate(H.parameters[1]),instantiate(Tuple{H.parameters[2:end]...})...)
	else
		return (instantiate(H.parameters[1]),)
	end
end