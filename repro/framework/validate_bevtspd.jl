# Independent validator wrapper around evaluate_bevtspd_route.
using ESNGSDM
using SHA

"""Corrected sortie-only drone_km (L→C→R Euclidean/energy matrix)."""
function corrected_drone_km(inst, truck_route::Vector{Int}, drone_route::Vector{Int})
    shared = [v for v in truck_route if v in Set(drone_route)]
    dp = Dict(v => i for (i, v) in enumerate(drone_route))
    dronekm = 0.0
    sorties = 0
    for (l, r) in zip(shared, shared[2:end])
        df = dp[l]
        dl = dp[r]
        if dl - df == 2
            c = drone_route[df + 1]
            sorties += 1
            dronekm += inst.drone_energy_matrix[l, c] + inst.drone_energy_matrix[c, r]
        elseif dl - df != 1
            error("bad sortie between $l and $r: drone span $(dl - df)")
        end
    end
    return dronekm, sorties
end

function corrected_truck_km(inst, truck_route::Vector{Int})
    return sum(inst.truck_energy_matrix[a, b] for (a, b) in zip(truck_route, truck_route[2:end]); init=0.0)
end

"""Validate a BEVTSPD route; returns NamedTuple with feasibility + metrics."""
function validate_bevtspd(inst, truck_route::Vector{Int}, drone_route::Vector{Int};
                          geometry_sha::AbstractString="",
                          expected_geometry_sha::AbstractString="")
    t0 = time_ns()
    ev = evaluate_bevtspd_route(inst, truck_route, drone_route)
    validate_s = (time_ns() - t0) / 1e9
    truck_km = corrected_truck_km(inst, truck_route)
    drone_km, sorties = corrected_drone_km(inst, truck_route, drone_route)
    station_visits = count(v -> v in Set(inst.station_nodes), truck_route)
    sha_ok = isempty(expected_geometry_sha) || (geometry_sha == expected_geometry_sha)
    customers = Set(inst.customer_nodes)
    served = Set(Int[])
    for v in truck_route
        v in customers && push!(served, v)
    end
    for v in drone_route
        v in customers && push!(served, v)
    end
    coverage_ok = served == customers
    return (
        feasible = ev.feasible && sha_ok && coverage_ok,
        evaluate_feasible = ev.feasible,
        coverage_ok = coverage_ok,
        geometry_sha_ok = sha_ok,
        makespan_h = ev.objective,
        truck_km = truck_km,
        drone_km = drone_km,
        sorties = sorties,
        station_visits = station_visits,
        validate_s = validate_s,
        detail = ev,
    )
end
