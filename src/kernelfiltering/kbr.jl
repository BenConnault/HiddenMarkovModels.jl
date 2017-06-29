probnorm(mu)=normalize(mu.*(mu.>0),1)

"""
    ksr2(kx,gx)

Compute the kernel sum rule based on a fixed sample ``(x,y)_{1:n}``.
`xx` has Gramian `kx`.
`gx[i]` gives the inner product of the input prior `mu` with ``k_{x_i}``. 
If `mu` is given with coordinates `b` in `xx`, then `g:=kx*b`.
Return the coordinates of the posterior in `yy`.
"""
function ksr2(kx,gx,tol=1.0)
    n=size(kx,2)
    (kx+tol/sqrt(n)*n*I)\gx    
end

"""
    kbr2(kx,ky,gx,gy)

Compute the kernel Bayes rule based on a fixed sample ``(x,y)_{1:n}``.
`xx` and `yy` have Gramians `kx` and `ky`.
`gx[i]` gives the inner product of the input prior `mu` with ``k_{x_i}``. 
If `mu` is given with coordinates `b` in `xx`, then `g:=kx*b`.
`gy[j]` gives the inner product of the data `k_y` with ``k_{y_j}``, 
ie. `gy[j]=k(yy[j],y)`. 
Return the coordinates of the posterior in `xx`.
"""
function kbr2(kx,ky,gx,gy,tol=1.0)
    mu=ksr2(kx,gx)
    n=length(mu)
    # `mu` is the coordinates the prior times conditional
    # ie both of the marginal on `y` in the y_j basis,
    # and of the joint in the (x,y)_j basis.
    dmu=diagm(mu)
    pix=dmu*ky*(((dmu*ky)^2+tol/sqrt(n)*I)\(dmu*gy))
    pix
end


"""
    kbr(q,ky,mux,gy)

Compute the kernel Bayes rule based on a precomputed transition function matrix from `xx` to `yy`.
`yy` has Gramian `ky`.
`mux` the coordinate vector of the prior in `xx`.
`gy[j]` gives the inner product of the data `k_y` with ``k_{y_j}``, 
ie. `gy[j]=k(yy[j],y)`. 
Return the coordinate vector of the posterior in `xx`.
"""
function kbr(q,ky,mux,gy,tol=1.0)
    # mu=diagm(mux)*q    # you don't want to store the full joint in memory, better to recompute it on the fly below
    muy=upq(mux,q)
    n=length(muy)
    dmuy=diagm(muy)
    pix=diagm(mux)*q*((ky*dmuy+tol/sqrt(n)*I)\gy)     
    probnorm(pix)
end

function kbrbis(q,ky,mux,gy,tol=1.0)
    # mu=diagm(mux)*q    # you don't want to store the full joint in memory, better to recompute it on the fly below
    muy=upq(mux,q)
    n=length(muy)
    dmuy=diagm(muy)
    pix=diagm(mux)*q*(ky\((ky*dmuy+tol/sqrt(n)*I)\gy))    #MYSTERIOUSLY WORKS  
    probnorm(pix)
end
