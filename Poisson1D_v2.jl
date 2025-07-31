using Plots, LinearAlgebra

function main()
    Lx  = 1.0
    ncx = 200
    Δx  = Lx / ncx
    xce = LinRange(-Lx/2 - Δx/2, Lx/2 + Δx/2, ncx+2)
    xv  = LinRange(-Lx/2, Lx/2, ncx+1)

    k0  = 1.0
    σ   = Lx /10

    BC  = (w = 1.0, e = 2.0)
    
    # Allocate Arrays
    u    = zeros(ncx+2)
    q    = zeros(ncx+1)
    k    = k0 * ones(ncx+1)
    r    = zeros(ncx)
    r0   = zeros(ncx)
    b    = zeros(ncx)
    ∂u∂τ = zeros(ncx)

    # Initialize fields
    b  .= 5 * exp.(-xce[2:end-1].^2 / σ^2)
    k  .= 1.0 .+ k0 * exp.(-xv.^2 / σ^2)

    # Iteration parameters
    CFL   = 0.98                                            # Courant-Friedrichs-Levi criterium
    cfact = 0.5
    λmax  = maximum(2 * (k[2:end] .+ k[1:end-1]) / Δx^2 )   # maximum eigenvalue
    λmin  = 0.0                                             # minimum eigenvalue
    Δτ    = 2/sqrt(λmax) * CFL                              # pseudo time step
    c     = 2 * sqrt(λmin) * cfact
    α     = (2 - c * Δτ) / (2 + c * Δτ)
    β     = 2 * Δτ / (2 + c * Δτ)

    niter = 2e5
    ϵ     = 1e-6

    for iter = 1:niter
        # Store previous residual
        r0 .= r

        # Set BC
        u[1]   = BC.w * 2  - u[2]
        u[end] = BC.e * 2 - u[end-1]

        # Evaluate q
        q .= -k .* diff(u)/Δx

        # Evaluate residual
        r .= - diff(q)/Δx .+ b

        # Update ∂u∂τ
        ∂u∂τ .= β .* r .+ α .* ∂u∂τ

        # Update u
        u[2:end-1] .+= Δτ .* ∂u∂τ

        if iter == 1 || mod(iter,1e3) == 0
            nr = norm(r)
            println(nr)
            if isnan(nr)
                error("norm(r) = NaN")
            end
            if nr < ϵ
                println(iter)
                break
            end

            #  Pseudo-Transient parameters
            λmax = 4 * maximum(k) / Δx^2
            λmin = abs.(((r .- r0)' * (∂u∂τ .* Δτ)) / ((∂u∂τ .* Δτ)' * (∂u∂τ .* Δτ)))
            Δτ   = 2/sqrt(λmax) * CFL
            c    = 2 * sqrt(λmin) * cfact
            α    = (2 - c * Δτ) / (2 + c * Δτ)
            β    = 2 * Δτ / (2 + c * Δτ)
        end
    end
    # Visualization
    p1 = plot(xce, u)
    p2 = plot(xce[2:end-1], b)
    display(plot(p1,p2))
    sleep(0.1)
end

main()