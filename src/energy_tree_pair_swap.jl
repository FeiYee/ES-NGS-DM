"""
Energy-aware pair-swap descent over ALL truck-only customer positions.
Moves can cross truck/drone synchronization rings, while fixed drone customer
assignments, launch/rendezvous nodes and charging station visits remain intact.

The changed driving arcs are replaced within a single contiguous action
interval. Every accepted move is rechecked with the full route verifier.
This is a deterministic diagnostic neighborhood, not a complete VNS.
"""
function improve_truck_only_pair_swaps(
    instance::BEVTSPDInstance,
    truck_route::AbstractVector{Int},
    drone_route::AbstractVector{Int};
    screening::Symbol = :tree,
    max_passes::Int = 8,
    time_budget_seconds::Real = 30.0,
    atol::Float64 = 1e-10,
)
    screening in (:tree, :direct) ||
        throw(ArgumentError("screening must be :tree or :direct"))
    max_passes >= 0 || throw(ArgumentError("max_passes must be nonnegative"))
    time_budget_seconds > 0 || throw(ArgumentError("time budget must be positive"))
    t0 = time_ns()
    current_truck = collect(truck_route)
    current_drone = collect(drone_route)
    original = evaluate_bevtspd_route(instance, current_truck, current_drone)
    if !original.feasible
        return TreeScreenedSearchResult(
            false, Inf, current_truck, current_drone, Inf, NaN, 0, 0, 0,
            (time_ns() - t0) * 1e-9, screening,
        )
    end
    current_objective = original.objective
    initial_objective = current_objective
    final_energy = original.simulation.final_energy
    passes = 0
    checked = 0
    feasible_count = 0
    station_set = Set(instance.station_nodes)
    customer_set = Set(instance.customer_nodes)
    shared = intersect(Set(current_truck), Set(current_drone))
    energy_parameters = instance.energy_parameters

    while passes < max_passes && (time_ns() - t0) * 1e-9 < time_budget_seconds
        positions = [
            i for i in 2:(length(current_truck) - 1)
            if current_truck[i] in customer_set && !(current_truck[i] in shared)
        ]
        length(positions) >= 2 || break

        actions = route_energy_actions(
            current_truck, current_drone,
            instance.truck_energy_matrix, instance.drone_energy_matrix;
            swap_stations = station_set,
            swap_service_time = instance.swap_service_time,
        )
        edge_action = findall(action -> action isa TruckDrive, actions)
        length(edge_action) == length(current_truck) - 1 ||
            error("inconsistent driving-edge mapping")
        energy_tree = screening == :tree ? EnergyTransferTree(actions, energy_parameters) : nothing
        edge_ring, ring_truck, ring_drone = _screened_ring_data(
            instance, current_truck, current_drone,
        )

        best_objective = current_objective
        best_pair = (0, 0)
        best_energy = NaN
        timed_out = false

        for p in 1:(length(positions) - 1)
            timed_out && break
            for q in (p + 1):length(positions)
                if (time_ns() - t0) * 1e-9 >= time_budget_seconds
                    timed_out = true
                    break
                end
                i, j = positions[p], positions[q]
                swapped = copy(current_truck)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                changed_edges = sort!(unique([i - 1, i, j - 1, j]))
                action_first = edge_action[first(changed_edges)]
                action_last = edge_action[last(changed_edges)]
                replacement = copy(actions[action_first:action_last])

                time_delta = Dict{Int, Float64}()
                for edge in changed_edges
                    x_old, y_old = current_truck[edge], current_truck[edge + 1]
                    x_new, y_new = swapped[edge], swapped[edge + 1]
                    replacement[edge_action[edge] - action_first + 1] = TruckDrive(
                        x_new, y_new, instance.truck_energy_matrix[x_new, y_new],
                    )
                    ring = edge_ring[edge]
                    delta = instance.truck_time_matrix[x_new, y_new] -
                        instance.truck_time_matrix[x_old, y_old]
                    time_delta[ring] = get(time_delta, ring, 0.0) + delta
                end

                checked += 1
                predicted_energy = nothing
                if screening == :tree
                    result = evaluate_energy_replacement(
                        energy_tree, action_first, action_last, replacement;
                        initial_energy = energy_parameters.truck_capacity,
                    )
                    result.feasible || continue
                    predicted_energy = result.final_energy
                else
                    candidate_actions = vcat(
                        actions[1:(action_first - 1)],
                        replacement,
                        actions[(action_last + 1):end],
                    )
                    result = simulate_energy(
                        candidate_actions, energy_parameters.truck_capacity,
                        energy_parameters,
                    )
                    result.feasible || continue
                    predicted_energy = result.final_energy
                end
                feasible_count += 1
                candidate_objective = current_objective
                for (ring, delta) in time_delta
                    candidate_objective += max(ring_truck[ring] + delta, ring_drone[ring]) -
                        max(ring_truck[ring], ring_drone[ring])
                end
                if candidate_objective < best_objective - atol
                    best_objective = candidate_objective
                    best_pair = (i, j)
                    best_energy = predicted_energy
                end
            end
        end
        best_pair[1] == 0 && break
        i, j = best_pair
        current_truck[i], current_truck[j] = current_truck[j], current_truck[i]
        verified = evaluate_bevtspd_route(instance, current_truck, current_drone)
        verified.feasible || error("pair swap accepted an energy-infeasible route")
        isapprox(verified.objective, best_objective; atol = 1e-8, rtol = 1e-8) ||
            error("pair-swap incremental time disagrees with full verifier")
        isapprox(verified.simulation.final_energy, best_energy; atol = 1e-8, rtol = 1e-8) ||
            error("pair-swap predicted state disagrees with full verifier")
        current_objective = verified.objective
        final_energy = verified.simulation.final_energy
        passes += 1
    end

    return TreeScreenedSearchResult(
        true, current_objective, current_truck, current_drone,
        initial_objective, final_energy, passes, checked, feasible_count,
        (time_ns() - t0) * 1e-9, screening,
    )
end

function improve_truck_only_pair_swaps(
    instance::BEVTSPDInstance,
    result::Union{EnergyICPResult, EnergyRoleSearchResult};
    kwargs...,
)
    return improve_truck_only_pair_swaps(
        instance, result.truck_route, result.drone_route; kwargs...,
    )
end
