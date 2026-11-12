"""
Feasibility-screened adjacent customer swaps inside a truck/drone synchronization
ring. Unlike the standalone transfer microbenchmark, every accepted move is
checked against the complete BEVTSPD objective and an independent route evaluator.

The :tree and :direct versions enumerate the same moves and use identical
objective calculations, differing only in energy feasibility screening.
This is a controlled integration experiment, not yet cross-epoch VLNS.
"""
struct TreeScreenedSearchResult
    feasible::Bool
    objective::Float64
    truck_route::Vector{Int}
    drone_route::Vector{Int}
    initial_objective::Float64
    final_energy::Float64
    passes::Int
    candidates_checked::Int
    feasible_candidates::Int
    elapsed_seconds::Float64
    screening::Symbol
end

function _screened_ring_data(
    instance::BEVTSPDInstance,
    truck_route::Vector{Int},
    drone_route::Vector{Int},
)
    station_set = Set(instance.station_nodes)
    drone_position = Dict(node => index for (index, node) in enumerate(drone_route))
    shared_positions = [
        index for (index, node) in enumerate(truck_route)
        if haskey(drone_position, node)
    ]
    edge_ring = zeros(Int, length(truck_route) - 1)
    ring_truck = Float64[]
    ring_drone = Float64[]
    for ring in 1:(length(shared_positions) - 1)
        first_pos, last_pos = shared_positions[ring], shared_positions[ring + 1]
        truck_time = 0.0
        for edge in first_pos:(last_pos - 1)
            truck_time += instance.truck_time_matrix[truck_route[edge], truck_route[edge + 1]]
            truck_route[edge + 1] in station_set && (truck_time += instance.swap_service_time)
            edge_ring[edge] = ring
        end
        launch, rendezvous = truck_route[first_pos], truck_route[last_pos]
        dfirst, dlast = drone_position[launch], drone_position[rendezvous]
        drone_time = 0.0
        if dlast - dfirst == 2
            customer = drone_route[dfirst + 1]
            drone_time = (launch == instance.start_depot ? 0.0 : instance.launch_time) +
                instance.drone_time_matrix[launch, customer] +
                instance.drone_time_matrix[customer, rendezvous] +
                instance.retrieve_time
        end
        push!(ring_truck, truck_time)
        push!(ring_drone, drone_time)
    end
    all(!=(0), edge_ring) || error("unassigned truck edge in synchronization rings")
    return edge_ring, ring_truck, ring_drone
end

function improve_same_ring_adjacent_swaps(
    instance::BEVTSPDInstance,
    truck_route::AbstractVector{Int},
    drone_route::AbstractVector{Int};
    screening::Symbol = :tree,
    max_passes::Int = 20,
    time_budget_seconds::Real = 30.0,
    atol::Float64 = 1e-10,
)
    screening in (:tree, :direct) ||
        throw(ArgumentError("screening must be :tree or :direct"))
    max_passes >= 0 || throw(ArgumentError("max_passes must be nonnegative"))
    time_budget_seconds > 0 || throw(ArgumentError("time_budget_seconds must be positive"))
    t0 = time_ns()
    current_truck = collect(truck_route)
    current_drone = collect(drone_route)
    initial = evaluate_bevtspd_route(instance, current_truck, current_drone)
    if !initial.feasible
        return TreeScreenedSearchResult(
            false, Inf, current_truck, current_drone, Inf, NaN, 0, 0, 0,
            (time_ns() - t0) * 1e-9, screening,
        )
    end

    current_objective = initial.objective
    start_objective = current_objective
    final_energy = initial.simulation.final_energy
    passes = 0
    checked = 0
    feasible = 0
    customer_set = Set(instance.customer_nodes)
    shared = Set(intersect(Set(current_truck), Set(current_drone)))
    parameters = instance.energy_parameters

    while passes < max_passes && (time_ns() - t0) * 1e-9 < time_budget_seconds
        actions = route_energy_actions(
            current_truck,
            current_drone,
            instance.truck_energy_matrix,
            instance.drone_energy_matrix;
            swap_stations = Set(instance.station_nodes),
            swap_service_time = instance.swap_service_time,
        )
        edge_actions = findall(action -> action isa TruckDrive, actions)
        length(edge_actions) == length(current_truck) - 1 ||
            error("truck edge/action mapping mismatch")
        tree = screening == :tree ? EnergyTransferTree(actions, parameters) : nothing
        edge_ring, ring_truck, ring_drone =
            _screened_ring_data(instance, current_truck, current_drone)
        reconstructed = sum(max(t, d) for (t, d) in zip(ring_truck, ring_drone))
        isapprox(reconstructed, current_objective; atol = 1e-8, rtol = 1e-8) ||
            error("ring objective does not match independent verifier")

        best_index = 0
        best_objective = current_objective
        best_predicted_energy = NaN
        for index in 2:(length(current_truck) - 2)
            (time_ns() - t0) * 1e-9 < time_budget_seconds || break
            a, b = current_truck[index], current_truck[index + 1]
            a in customer_set && b in customer_set || continue
            a in shared && continue
            b in shared && continue
            left, right = current_truck[index - 1], current_truck[index + 2]
            start_action, end_action = edge_actions[index - 1], edge_actions[index + 1]
            end_action == start_action + 2 ||
                error("noncontiguous actions in adjacent customer swap")
            ring = edge_ring[index - 1]
            ring == edge_ring[index] == edge_ring[index + 1] ||
                error("swap crosses a synchronization boundary")

            replacements = AbstractEnergyAction[
                TruckDrive(left, b, instance.truck_energy_matrix[left, b]),
                TruckDrive(b, a, instance.truck_energy_matrix[b, a]),
                TruckDrive(a, right, instance.truck_energy_matrix[a, right]),
            ]
            checked += 1
            predicted_energy = nothing
            if screening == :tree
                result = evaluate_energy_replacement(
                    tree, start_action, end_action, replacements;
                    initial_energy = parameters.truck_capacity,
                )
                result.feasible || continue
                predicted_energy = result.final_energy
            else
                candidate_actions = vcat(
                    actions[1:(start_action - 1)],
                    replacements,
                    actions[(end_action + 1):end],
                )
                result = simulate_energy(candidate_actions, parameters.truck_capacity, parameters)
                result.feasible || continue
                predicted_energy = result.final_energy
            end
            feasible += 1

            old_time = instance.truck_time_matrix[left, a] +
                instance.truck_time_matrix[a, b] +
                instance.truck_time_matrix[b, right]
            new_time = instance.truck_time_matrix[left, b] +
                instance.truck_time_matrix[b, a] +
                instance.truck_time_matrix[a, right]
            proposed_truck = ring_truck[ring] + (new_time - old_time)
            candidate_objective = current_objective -
                max(ring_truck[ring], ring_drone[ring]) +
                max(proposed_truck, ring_drone[ring])
            if candidate_objective < best_objective - atol
                best_objective = candidate_objective
                best_index = index
                best_predicted_energy = predicted_energy
            end
        end
        best_index == 0 && break
        current_truck[best_index], current_truck[best_index + 1] =
            current_truck[best_index + 1], current_truck[best_index]
        verified = evaluate_bevtspd_route(instance, current_truck, current_drone)
        verified.feasible || error("screened search accepted an infeasible route")
        isapprox(verified.objective, best_objective; atol = 1e-8, rtol = 1e-8) ||
            error("incremental objective differs from verifier")
        isapprox(verified.simulation.final_energy, best_predicted_energy;
                 atol = 1e-8, rtol = 1e-8) ||
            error("incremental final SoC differs from verifier")
        current_objective = verified.objective
        final_energy = verified.simulation.final_energy
        passes += 1
    end

    return TreeScreenedSearchResult(
        true, current_objective, current_truck, current_drone, start_objective,
        final_energy, passes, checked, feasible,
        (time_ns() - t0) * 1e-9, screening,
    )
end

function improve_same_ring_adjacent_swaps(
    instance::BEVTSPDInstance,
    result::Union{EnergyICPResult, EnergyRoleSearchResult};
    kwargs...,
)
    return improve_same_ring_adjacent_swaps(
        instance, result.truck_route, result.drone_route; kwargs...,
    )
end
