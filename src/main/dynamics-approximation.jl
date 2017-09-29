# Approximation using density evaluations ("cheating")
# You need uniformly spaced grids.
function markovapprox(model,subkernel,bxx,byy)
    nx=length(bxx)
    ny=length(byy)
    q=zeros(nx,ny)
    for ix=1:nx
        rho = 0.0
        for jy=1:ny
            q[ix,jy] = cpdf(model,subkernel,bxx[ix],byy[jy])
            rho += q[ix,jy]
        end
        q[ix,:] /= rho
    end
    q
end


# Approximation using simulation and kernel projections
function markovapprox(model,subkernel,bxx,byy,ky,m=1000,tol=0.1)
    nx=length(bxx)
    ny=length(byy)
    q=zeros(nx,ny)
    for ix=1:nx
        print("$ix ")
        gi=zeros(ny)
        for j=1:m
            y=rand(model,subkernel,bxx[ix])
            for jy=1:ny
                gi[jy]=gi[jy]+kk(byy[jy],y)/m
            end            
        end
        q[ix,:]=probnorm((ky+tol*I/sqrt(ny))\gi)
    end
    q
end


# TYPE DESIGN NOTE
# A given model can have several subkernels.
# Eg. a StrictHMM has transitions x-> x' and measurements x' -> y'.
# We need a mechanism to say which subkernel we are currently approximating.
# In a previous iteration model types like StrictHMM had one field of type MarkovKernel 
# for each subkernel eg:
# The current system with a `subkernel` flag seems leaner.
##################