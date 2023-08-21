using LinearAlgebra
import LazySets: convex_hull
import Combinatorics: combinations

function numerical_range(A::Matrix, resolution::Number = 0.01)
    w = ComplexF64[]
    for θ = 0:resolution:2pi
        Ath = exp(1im * -θ) * A
        Hth = (Ath + Ath') / 2
        F = eigen(Hth)
        m = F.values[end]
        s = findall(≈(m), F.values)
        if length(s) == 1
            p = F.vectors[:, s]' * A * F.vectors[:, s]
            push!(w, tr(p))
        else
            Kth = 1im * (Hth - Ath)
            pKp = F.vectors[:, s]' * Kth * F.vectors[:, s]
            FF = eigen(pKp)
            mm = FF.values[1]
            ss = findall(≈(mm), FF.values)
            p =
                FF.vectors[:, ss[1]]' *
                F.vectors[:, s]' *
                A *
                F.vectors[:, s] *
                FF.vectors[:, ss[1]]
            push!(w, tr(p))
            mM = maximum(FF.values[end])
            sS = findall(≈(mM), FF.values)
            p =
                FF.vectors[:, sS[1]]' *
                F.vectors[:, s]' *
                A *
                F.vectors[:, s] *
                FF.vectors[:, sS[1]]
            push!(w, tr(p))
        end
    end
    nr = convex_hull(collect.(collect(zip(real(w), imag(w)))))
    return reduce(hcat, nr)'
end

side(point, v1, v2) = (point[1] - v2[1]) * (v1[2] - v2[2]) - (v1[1] - v2[1]) * (point[2] - v2[2])

function in_triangle(point, triangle)
    side_tests = [side(point, vertices...) for vertices in combinations(triangle, 2)]
    # all(x->x>0, side_tests)
    all(x->x<0, side_tests)
end

function circle(r::Real, n::Int, c::Vector{<:Real} = [0., 0.])
    t = LinRange(0, 2π, n)
    x = c[1] .+ r * cos.(t)
    y = c[2] .+ r * sin.(t)
    collect.(eachrow(hcat(x, y)))
end

circle(r::Real, n::Int, c::ComplexF64) = circle([real(c), imag(c)], r, n)

function ellipse(f1, f2, e, n)
    t = LinRange(0, 2π, n)
    z = (f1-f2)/2;
    c = abs(z);
    r = angle(z);
    a = c/e;
    b = sqrt(a^2 - c^2);
    z1 = (f1+f2)/2;
    p = real(z1);
    q = imag(z1);

    x = a * cos.(r) * cos.(t) .- b * sin(r) * sin.(t) .+ p
    y = a * sin.(r) * cos.(t) .+ b * cos(r) * sin.(t) .+ q

    collect.(eachrow(hcat(x, y)))
end

function subrange(q::Real, n::Int, evs::Vector)
    @assert length(evs) == 3
    qevs = q * evs
    points = Vector{Float64}[]
    for comb in combinations(qevs, 2)
        append!(points, ellipse(comb[1], comb[2], q, n))
    end
    small_circle = circle(q, n)
    unit_circle = circle(1, n)
    triangle = collect.(eachrow(hcat(real.(evs), imag(evs))))
    mask = in_triangle.(small_circle, Ref(triangle))
    for (i, m) in enumerate(mask)
        m && push!(points, unit_circle[i])
    end
    return points
end

function qrange(A, q::Real=1, n::Int=1000)
    q == 1 && return numerical_range(A)
    @assert A' * A ≈ I "Only unitary matrices supported"
    evs = unique(eigvals(A))
    length(evs) == 2 && return convex_hull(ellipse(q*evs[1], q*evs[2], q, n))
    points = Vector{Float64}[]
    for comb in combinations(evs, 3)
        sort!(comb, by=angle)
        append!(points, subrange(q, n, comb))
    end
    return reduce(hcat, convex_hull(points))'
end