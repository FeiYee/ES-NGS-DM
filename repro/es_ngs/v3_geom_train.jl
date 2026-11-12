# Train-only V3-style geometry (same generator as freeze_structured_v3.jl).
# Seeds MUST be outside frozen Dev/Holdout range 20261011–20261040.
using Random

const LIM = 20.0

function demand_coords(n, seed, profile)
    rng = MersenneTwister(seed + n * 3001 + (profile == "Q" ? 13 : profile == "R" ? 31 : 59))
    x = zeros(n, 2)
    for i in 1:n
        if profile == "Q"
            if rand(rng) < 0.10
                x[i, :] .= 40 .* rand(rng, 2) .- 20
            else
                cx = rand(rng, Bool) ? 14.0 : -14.0
                cy = rand(rng, Bool) ? 14.0 : -14.0
                x[i, 1] = clamp(cx + 2.0 * randn(rng), -LIM, LIM)
                x[i, 2] = clamp(cy + 2.0 * randn(rng), -LIM, LIM)
            end
        elseif profile == "R"
            if rand(rng) < 0.10
                x[i, :] .= 40 .* rand(rng, 2) .- 20
            else
                a = 2pi * rand(rng)
                rad = 14.0 + 5.0 * rand(rng)
                x[i, 1] = rad * cos(a)
                x[i, 2] = rad * sin(a)
            end
        elseif profile == "L"
            if rand(rng) < 0.10
                x[i, :] .= 40 .* rand(rng, 2) .- 20
            else
                x[i, 1] = clamp((rand(rng, Bool) ? 13.0 : -13.0) + 2.0 * randn(rng), -LIM, LIM)
                x[i, 2] = 36.0 * rand(rng) - 18.0
            end
        else
            error("unknown demand profile")
        end
    end
    return x
end

function station_coords(k, n, seed, profile)
    rng = MersenneTwister(seed + n * 3007 + (profile == "C" ? 71 : 97))
    x = zeros(k, 2)
    for j in 1:k
        if rand(rng) < 0.25
            x[j, :] .= 40 .* rand(rng, 2) .- 20
        elseif profile == "C"
            x[j, :] .= 10 .* rand(rng, 2) .- 5
        elseif profile == "P"
            ang = 2pi * rand(rng)
            rad = 14.0 + 5.0 * rand(rng)
            x[j, 1] = rad * cos(ang)
            x[j, 2] = rad * sin(ang)
        else
            error("unknown station profile")
        end
    end
    return x
end

function make_train_raw(n::Int, k::Int, seed::Int, demand::AbstractString, supply::AbstractString)
    (20261011 <= seed <= 20261040) && error("train seed collides with frozen Dev/Holdout: $seed")
    customers = demand_coords(n, seed, demand)
    stations = station_coords(k, n, seed, supply)
    return (n=n, k=k, seed=seed, split="train", demand=demand, supply=supply,
            customers=customers, stations=stations, version="es_ngs_train_v1")
end
