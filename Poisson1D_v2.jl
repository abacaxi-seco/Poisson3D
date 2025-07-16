using Plots, LinearAlgebra

function main()
    k0 = 1.0
    σ  = Lx /10

    Lx = 1.0
    ncx = 200
    Δx = Lx / ncx
    xce= LinRange(-Lx/2 - Δx/2, Lx/2 + Δx/2, ncx+2)
    xv = LinRange(-Lx/2, Lx/2, ncx+1)

    BC = (w = 1.0, e = 2.0)
    
    # Allocate Arrays
    u = zeros(ncx+2)
    q = zeros(ncx+1)
    k = k0 * ones(ncx+1)
    r = zeros(ncx)
    b = zeros(ncx)
    ∂u∂τ = zeros(ncx)

    # Initialize fields
    b  .= 5 * exp.(-xce[2:end-1].^2 / σ^2)
    k  .= 5 * exp.(-xv.^2 / σ^2)

    # Iteration parameters
    Δτ    = Δx^2 /k0 / 5    # pseudo time step
    θ     = 0.001
    niter = 2e5
    ϵ     = 1e-6

    for iter = 1:niter
        # set BC
        u[1]   = BC.w * 2  - u[2]
        u[end] = BC.e * 2 - u[end-1]

        #evaluate q
        q .= -k .* diff(u)/Δx

        # evaluate residual
        r .= - diff(q)/Δx .+ b

        # update ∂u∂τ
        ∂u∂τ .= r .+ (1-θ) .* ∂u∂τ

        # update u
        u[2:end-1] .+= Δτ .* ∂u∂τ

        if mod(iter,1e3) == 0
            nr = norm(r)
            println(nr)
            if isnan(nr)
                error("some error")
            end
            if nr < ϵ
                println(iter)
                break
            end
        end
    end

    p1 = plot(xce, u)
    p2 = plot(xce[2:end-1], b)
    display(plot(p1,p2))
    sleep(0.1)
end

main()