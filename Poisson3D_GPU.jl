using Plots, ParallelStencil, WriteVTK, Printf

const USE_GPU = false       # boolean to select if code runs on GPU or CPU
const GPU_ID  = 0           # select the ID of the GPU in a multi-GPU system

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);    # runs parallel on CPU cores
end

@parallel_indices (i,j,k) function InitialCondition!(u, c, r) # kernel automatically runs as a loop for the three indices i, j, k
    if (c.x[i].^2 .+ c.y[j].^2 .+ c.z[k].^2) < r^2
        u[i,j,k] = 5.0
    else
        u[i,j,k] = 1.0
    end
    return nothing
end

@parallel_indices (i,j,k) function ComputeFluxes!(q, u, D, Δ)
    if i <= size(q.x, 1) && j <= size(q.x, 2) && k <= size(q.x, 3)
        q.x[i, j, k] = - D.x[i, j, k] * (u[i+1, j, k] - u[i, j, k]) / Δ.x
    end

    if i <= size(q.y, 1) && j <= size(q.y, 2) && k <= size(q.y, 3)
        q.y[i, j, k] = - D.y[i, j, k] * (u[i, j+1, k] - u[i, j, k]) / Δ.y
    end

    if i <= size(q.z, 1) && j <= size(q.z, 2) && k <= size(q.z, 3)
        q.z[i, j, k] = - D.z[i, j, k] * (u[i, j, k+1] - u[i, j, k]) / Δ.z
    end
    return nothing
end

@parallel_indices (i,j,k) function Update_u!(u, q, Δ, Δt)
    # a lot of indices are shifted here to avoid the ghost cells
    if i <= size(u, 1)-2 && j <= size(u, 2)-2 && k <= size(u, 3)-2
        ∂q∂x = (q.x[i+1, j+1, k+1] - q.x[i, j+1, k+1]) / Δ.x
        ∂q∂y = (q.y[i+1, j+1, k+1] - q.y[i+1, j, k+1]) / Δ.y
        ∂q∂z = (q.z[i+1, j+1, k+1] - q.z[i+1, j+1, k]) / Δ.z

        u[i+1, j+1, k+1] -= Δt * (∂q∂x + ∂q∂y + ∂q∂z)
    end
    return nothing
end

function main()
    D0 = 1.0

    L  = (x = 1, y = 1, z = 1)
    nc = (x= 50, y= 50, z= 50)
    Δ  = (x = L.x / nc.x, y = L.y / nc.y, z = L.z / nc.z)

    c  = (
        x = LinRange(-L.x/2 - Δ.x, L.x/2 + Δ.x, nc.x+2),
        y = LinRange(-L.y/2 - Δ.y, L.y/2 + Δ.y, nc.y+2),
        z = LinRange(-L.z/2 - Δ.z, L.z/2 + Δ.z, nc.z+2)
        )

    #= Allocate Arrays
    @zeros is a ParallelStencil macro to allocate the memory depending on used GPU =#
    u = @zeros(nc.x+2, nc.y+2, nc.z+2) 
    q = (
        x = @zeros(nc.x+1, nc.y+2, nc.z+2),
        y = @zeros(nc.x+2, nc.y+1, nc.z+2),
        z = @zeros(nc.x+2, nc.y+2, nc.z+1)
    )
    D = (
        x = D0 .* @ones(nc.x+1, nc.y+2, nc.z+2),
        y = D0 .* @ones(nc.x+2, nc.y+1, nc.z+2),
        z = D0 .* @ones(nc.x+2, nc.y+2, nc.z+1)
    )

    # Initial Condition
    r = 0.1   # radius of the sphere
    # launch kernel to set initial values of u
    @parallel InitialCondition!(u, c, r)

    # Time
    # min(Δ...) or minimum(Δ)
    Δt = min(Δ...)^2 / D0 / 6.1
    nt = 1e2

    for it = 1:nt
        # set BC (or don't...?)
        # compute q
        @parallel ComputeFluxes!(q, u, D, Δ)
        # update u
        @parallel Update_u!(u, q, Δ, Δt)

        # Visualization and Output as VTK
        if mod(it, 10) == 0
            p1 = heatmap(c.x, c.y, u[:,:, Int64(round(nc.z/2))], aspect_ratio=1, xlims=(-L.x/2,L.x/2))
        display(plot(p1))

            filename = @sprintf("Output%04d", it)
            vtk_grid(filename, c.x, c.y, c.z) do vtk
                vtk["u"] = u
            end
        end
    end
end

main()