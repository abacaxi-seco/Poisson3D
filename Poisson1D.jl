using Plots, LinearAlgebra

function main()
    k0 = 1.0
    
    Lx = 1.0
    ncx = 50
    Δx  = Lx / ncx
    xce = LinRange(-Lx/2 - Δx/2, Lx/2 + Δx/2, ncx+2)
    BC  = (w = 1.0, e = 2.0)

    # Allocate Arrays
    u = zeros(ncx+2)
    q = zeros(ncx+1)
    k = k0 * ones(ncx+1)
    r = zeros(ncx)
    b = zeros(ncx)

    # Iteration
    Δτ = Δx^2 /k0 / 2.1
    niter = 4e4

    for iter = 1:niter
        # set BC
        u[1]   = BC.w * 2  - u[2]
        u[end] = BC.e * 2 - u[end-1]

        # evaluate q
        q .= -k .* diff(u)/Δx

        # evaluate thee residual
        r .= - diff(q)/Δx .+ b

        # update u
        u[2:end-1] .+= Δτ .* r

        # Output
        if mod(iter,1e3) ==0
        nr = norm(r)
        println(nr)
        p1 = plot(xce, u)
        p2 = plot(xce[2:end-1], r)
        display(plot(p1,p2))
        sleep(0.1)
        end
    end
end

main()