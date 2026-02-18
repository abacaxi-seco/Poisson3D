using Plots, LinearAlgebra, StaticArrays
using Enzyme

function compute_residual_local(u, b, BC, Δx, k, len_u, i)
    uC = u[1]
    uW = i== 2    ? BC.w * 2  - uC : u[2]
    uE = i== len_u[1]-1 ? BC.e * 2 - uC : u[3]

    qW = -k[1] * (uC - uW) / Δx
    qE = -k[2] * (uE - uC) / Δx

    return -((qE - qW) / Δx + b)
end

function compute_residual(u, b, r0, BC, Δx, k, deriv, i)
    uC = u[i]
    uW = u[i-1]
    uE = u[i+1]
    𝑢 = @SVector[uC,uW,uE]
    kW = k[i-1]
    kE = k[i]
    k_l = @SVector[kW,kE]
    if deriv==true
        ∂r∂u = gradient(Reverse, compute_residual_local, 𝑢, Const(b[i]), Const(BC), Const(Δx), Const(k_l), Const(size(u)), Const(i))
        D = abs.(∂r∂u[1][1]) # diagonal entries of ∂r∂u
        G = sum(abs.(∂r∂u[1]))
        out = D, G
    else
        r = compute_residual_local(𝑢, b[i], BC, Δx, k_l, size(u), i)
        out = r, r0[i]
    end
    return out
end
# function update_residual(u, b, r, r0, Δx, BC, k, deriv)
#     for i = 2:size(u)[1]-1
#         result = compute_residual(u, b, r, Δx, BC, k, deriv, i)
#         r[i-1] = result[1]
#         r0[i-1] = result[2]
#     end
# end
function update_field(a, b, func, args)
    for i = 2:size(a)[1]-1 # skip ghosts
        result = func(args..., i)
        a[i] = result[1]
        b[i] = result[2]
    end
end

function update_u(u, ∂u∂τ, r, Δτ, D, β, α, i)
    ∂u∂τ_new = r[i] / D[i] + α * ∂u∂τ[i]
    u_new = u[i] + Δτ * β * ∂u∂τ_new
    return u_new, ∂u∂τ_new
end

function main()
    Lx  = 1.0
    ncx = 200
    nce = ncx + 2
    Δx  = Lx / ncx
    xce = LinRange(-Lx/2 - Δx/2, Lx/2 + Δx/2, ncx+2)
    xv  = LinRange(-Lx/2, Lx/2, ncx+1)

    k0  = 1.0
    σ   = Lx /10

    BC  = (w = 1.0, e = 2.0)
    
    # Allocate Arrays
    u    = zeros(nce)
    q    = zeros(ncx+1)
    k    = k0 * ones(ncx+1)
    r    = zeros(nce)
    r0   = zeros(nce)
    b    = zeros(nce)
    ∂u∂τ = zeros(nce)
    G    = ones(nce)
    D    = ones(nce)

    # Initialize fields
    b  .= 5 * exp.(-xce.^2 / σ^2)
    k  .= 1.0 .+ k0 * exp.(-xv.^2 / σ^2)

    # Evaluate G and D (Gershgorin)
    update_field(D, G, compute_residual, (u, b, r, BC, Δx, k, true))

    # Iteration parameters
    CFL   = 0.98                                            # Courant-Friedrichs-Levi criterium
    cfact = 0.5
    # λmax  = maximum(2 * (k[2:end] .+ k[1:end-1]) / Δx^2 )   # maximum eigenvalue
    λmax  = maximum(G ./ D)
    λmin  = 0.0                                             # minimum eigenvalue
    Δτ    = 2/sqrt(λmax) * CFL                              # pseudo time step
    c     = 2 * sqrt(λmin) * cfact
    α     = (2 - c * Δτ) / (2 + c * Δτ)
    β     = 2 * Δτ / (2 + c * Δτ)

    niter = 2e5
    ϵ     = 1e-6

    for iter = 1:niter
        # update resuidual r and r0
        update_field(r, r0, compute_residual, (u, b, r, BC, Δx, k, false))

        # Update u and ∂u∂τ
        update_field(u, ∂u∂τ, update_u, (u, ∂u∂τ, r, Δτ, D, β, α))

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

            update_field(D, G, compute_residual, (u, b, r, BC, Δx, k, true))

            #  Pseudo-Transient parameters
            λmax = maximum(G ./ D)
            λmin = abs.(sum((r .- r0) ./D .* (∂u∂τ .* Δτ))) / sum((∂u∂τ .* Δτ).^2)
            Δτ   = 2/sqrt(λmax) * CFL
            c    = 2 * sqrt(λmin) * cfact
            α    = (2 - c * Δτ) / (2 + c * Δτ)
            β    = 2 * Δτ / (2 + c * Δτ)
        end
    end
    # Visualization
    p1 = plot(xce[2:end-1], u[2:end-1], label="u", title="Poisson 1D with DR and AD")
    p2 = plot(xce, b, label="b")
    display(plot(p1,p2))
    sleep(0.1)
end

main()