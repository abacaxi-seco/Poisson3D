using Plots, LinearAlgebra, StaticArrays
using Enzyme
function av(x,y)
    return 0.5 * (x + y)
end

conductivity(abs_u, k0, a, n) = k0 + a * abs_u^n 

function compute_residual_local(u, b, BC, Δx, k, len_u, i)
    uC = u[1]
    uW = i== 2          ? (2*BC.w - uC) : u[2]
    uE = i== len_u[1]-1 ? (2*BC.e - uC) : u[3]

    k0, a, n = 1.0, 5e-3, 3
    kW = k[i-1] = conductivity(av(abs(uC), abs(uW)), k0, a, n)
    kE = k[i]   = conductivity(av(abs(uC), abs(uE)), k0, a, n)

    if kW <= 0 || kE <= 0 
        error("Negative diffusivity: kW = $kW, kE = $kE at i = $i, uC = $uC, uW = $uW, uE = $uE, u = $u")
    elseif !isfinite(kW) || !isfinite(kE)
        error("Non-finite diffusivity: kW = $kW, kE = $kE at i = $i, uC = $uC, uW = $uW, uE = $uE, u = $u")
    end

    qW = -kW * (uC - uW) / Δx
    qE = -kE * (uE - uC) / Δx

    return -((qE - qW) / Δx + b)
end

function compute_residual(u, b, r0, BC, Δx, k, deriv, i)
    uC = u[i]
    uW = u[i-1]
    uE = u[i+1]
    𝑢 = @SVector[uC,uW,uE]
    if deriv==true
        ∂r∂u = gradient(Reverse, compute_residual_local, 𝑢, Const(b[i]), Const(BC), Const(Δx), Const(k), Const(size(u)), Const(i))
        D    = abs.(∂r∂u[1][1]) # diagonal entries of ∂r∂u
        G    = sum(abs.(∂r∂u[1]))
        out  = D, G
    else
        r   = compute_residual_local(𝑢, b[i], BC, Δx, k, size(u), i)
        out = r, r0[i]
    end
    return out
end

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

    ∈
    # Initialize fields
    b  .= 5 * exp.(-xce.^2 / σ^2)
    # k  .= 1.0 .+ k0 * exp.(-xv.^2 / σ^2)
    # u  .= 1.0 .+ exp.(-xce.^2 / σ^2)

    # Evaluate G and D (Gershgorin)
    update_field(D, G, compute_residual, (u, b, r, BC, Δx, k, true))

    # Iteration parameters
    CFL   = 0.989                                           # Courant-Friedrichs-Levi criterium
    cfact = 0.5
    λmax  = maximum(G ./ D)                                 # maximum eigenvalue
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
            # Visualization
            p1 = plot(xce[2:end-1], u[2:end-1], ylabel="u", xlabel="x", label=:none, title="Iteration $iter")
            p2 = plot(xv, k, ylabel="k", xlabel="x", label=:none)
            p3 = plot(xce, b, ylabel="b", xlabel="x", label=:none)
            display(plot(p1,p2,p3))
            sleep(0.1)

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
    p1 = plot(xce[2:end-1], u[2:end-1], ylabel="u", xlabel="x", label=:none)
    p2 = plot(xce, b, ylabel="b", xlabel="x", label=:none)
    display(plot(p1,p2))
    sleep(0.1)
end

main()