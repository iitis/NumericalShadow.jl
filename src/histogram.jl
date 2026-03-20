using KernelAbstractions
using Atomix

function get_bounding_box(A::AbstractMatrix, q::Real = 1)
    nr = qrange(A, q)
    minimum(nr[:, 1]), maximum(nr[:, 1]), minimum(nr[:, 2]), maximum(nr[:, 2])
end

function get_bin_edges(A::AbstractMatrix, nbins_x::Int, q::Real = 1, nbins_y::Int = nbins_x)
    nbins_x > 0 || throw(ArgumentError("`nbins_x` must be positive, got $nbins_x"))
    nbins_y > 0 || throw(ArgumentError("`nbins_y` must be positive, got $nbins_y"))

    min_x, max_x, min_y, max_y = get_bounding_box(A, q)
    if min_x ≈ max_x
        min_x -= 0.5
        max_x += 0.5
    end
    if min_y ≈ max_y
        min_y -= 0.5
        max_y += 0.5
    end
    x_edges = min_x:(max_x-min_x)/nbins_x:max_x
    y_edges = min_y:(max_y-min_y)/nbins_y:max_y
    x_edges, y_edges
end

"""
    Hist2D(x_edges, y_edges)

2D histogram container used by shadow routines.

`x_edges` and `y_edges` define bin edges, while `hist` stores counts.
Additional optional fields (`nr`, `evs`, `other_range`) are metadata helpers
used by plotting/analysis workflows.
"""
mutable struct Hist2D
    x_edges::AbstractVector
    y_edges::AbstractVector
    hist::AbstractMatrix
    nr::AbstractMatrix
    evs::AbstractVector
    other_range::AbstractMatrix
    Hist2D(x_edges, y_edges, hist) = new(x_edges, y_edges, hist)
end

function Hist2D(x_edges::AbstractVector, y_edges::AbstractVector)
    backend = KernelAbstractions.get_backend(x_edges)
    hist = KernelAbstractions.zeros(backend, Int32, length(x_edges) - 1, length(y_edges) - 1)
    Hist2D(x_edges, y_edges, hist)
end

function Base.:+(h1::Hist2D, h2::Hist2D)
    Hist2D(h1.x_edges, h1.y_edges, h1.hist + h2.hist)
end

"""
    save(h::Hist2D, fname)

Serialize histogram `h` to an HDF5 file.
"""
function save(h::Hist2D, fname::String)
    HDF5.h5open(fname, "w") do io
        io["x_edges"] = Array(h.x_edges)
        io["y_edges"] = Array(h.y_edges)
        io["hist"] = Array(h.hist)

        if isdefined(h, :nr)
            io["nr"] = Array(h.nr)
        end
        if isdefined(h, :evs)
            io["evs"] = Array(h.evs)
        end
        if isdefined(h, :other_range)
            io["other_range"] = Array(h.other_range)
        end
    end
end

@kernel function histogram_kernel(xs, ys, @Const(x_edges), @Const(y_edges), hist_vec, nx_i32, ny_i32)
    i = @index(Global, Linear)
    if i <= length(xs)
        x = xs[i]
        y = ys[i]
        
        min_x = x_edges[1]
        step_x = x_edges[2] - x_edges[1]
        min_y = y_edges[1]
        step_y = y_edges[2] - y_edges[1]
        
        idx_x = trunc(Int32, (x - min_x) / step_x) + Int32(1)
        idx_y = trunc(Int32, (y - min_y) / step_y) + Int32(1)
        
        if idx_x > nx_i32 && idx_x <= nx_i32 + Int32(1)
            idx_x = nx_i32
        end
        if idx_y > ny_i32 && idx_y <= ny_i32 + Int32(1)
            idx_y = ny_i32
        end
        
        if Int32(1) <= idx_x <= nx_i32 && Int32(1) <= idx_y <= ny_i32
             lidx = idx_x + (idx_y - Int32(1)) * nx_i32
             @inbounds KernelAbstractions.@atomic hist_vec[lidx] += Int32(1)
        end
    end
end

function histogram(xs::AbstractVector, ys::AbstractVector, x_edges::AbstractVector, y_edges::AbstractVector)
    backend = KernelAbstractions.get_backend(xs)
    nx = length(x_edges) - 1
    ny = length(y_edges) - 1
    hist = KernelAbstractions.zeros(backend, Int32, nx, ny)
    
    if backend isa CPU
        # Workaround for Atomix on CPU if KA @atomic is broken
        # We use a manual loop for now for robustness on 1.11/1.12
        for i in 1:length(xs)
            x = xs[i]
            y = ys[i]
            min_x = x_edges[1]
            step_x = x_edges[2] - x_edges[1]
            min_y = y_edges[1]
            step_y = y_edges[2] - y_edges[1]
            idx_x = floor(Int, (x - min_x) / step_x) + 1
            idx_y = floor(Int, (y - min_y) / step_y) + 1
            nx_l = length(x_edges) - 1
            ny_l = length(y_edges) - 1
            if idx_x > nx_l && idx_x <= nx_l + 1; idx_x = nx_l; end
            if idx_y > ny_l && idx_y <= ny_l + 1; idx_y = ny_l; end
            if 1 <= idx_x <= nx_l && 1 <= idx_y <= ny_l
                @inbounds Atomix.@atomic hist[idx_x, idx_y] += Int32(1)
            end
        end
    else
        kernel = histogram_kernel(backend)
        kernel(xs, ys, x_edges, y_edges, vec(hist), Int32(nx), Int32(ny), ndrange=length(xs))
    end
    
    Hist2D(x_edges, y_edges, hist)
end

function histogram(points::AbstractVector{<:Complex}, x_edges, y_edges)
    histogram(real(points), imag(points), x_edges, y_edges)
end
