import LazySets: convex_hull

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

function circle_boundary(r::Real, n::Int, c::Vector{Real} = [0., 0.])
    t = LinRange(0, 2π, n)
    x = c[1] .+ r * cos.(t)
    y = c[2] .+ r * sin.(t)
    return x, y
end

circle_boundary(r::Real, n::Int, c::ComplexF64) = circle_boundary([real(c), imag(c)], r, n)

function ellipse_boundary(f1, f2, e, n)
    t = LinRange(0, 2π, n)
    z = (f1-f2)/2;
    c = abs(z);
    r = angle(z);
    a = c/e;
    b = sqrt(a^2 - c^2);
    z1 = (u+v)/2;
    p = real(z1);
    q = imag(z1);

    x = a * cos.(r) * cos.(t) .- b * sin(r) * sin(t) .+ p
    y = a * sin.(r) * cos.(t) .+ b * cos(r) * sin(t) .+ q

    return x, y
end

function qrange(A, q::Real=1)
    q == 1 && return numerical_range(A)


end