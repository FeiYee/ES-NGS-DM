"""Energy-aware result built from an ICP route and exact station repair."""
struct EnergyICPResult
    feasible::Bool
    objective::Float64
    truck_route::Vector{Int}
    drone_route::Vector{Int}
    final_energy::Float64
    base_objective::Float64
end

"""Result of deterministic energy-coupled truck/drone role descent."""
struct EnergyRoleSearchResult
    feasible::Bool
    objective::Float64
    truck_route::Vector{Int}
    drone_route::Vector{Int}
    final_energy::Float64
    iterations::Int
    candidates_evaluated::Int
end

struct _RepairLabel
    energy::Float64
    objective::Float64
    truck_route::Vector{Int}
end

function _prune_repair_labels(labels::Vector{_RepairLabel}; atol::Float64 = 1e-10)
    kept = _RepairLabel[]
    for candidate in sort(labels; by = label -> (label.objective, -label.energy))
        any(
            incumbent.objective <= candidate.objective + atol &&
            incumbent.energy + atol >= candidate.energy for incumbent in kept
        ) && continue
        filter!(kept) do incumbent
            !(candidate.objective <= incumbent.objective + atol &&
              candidate.energy + atol >= incumbent.energy)
        end
        push!(kept, candidate)
    end
    return kept
end

"""
Find the optimal charging-station paths for a fixed truck-customer order and
fixed drone sorties.

Customer order and synchronization boundaries are not changed. Under the
EVTSPD-FF contract that every original-network edge fits within one full truck
battery, default repair considers a direct path or one station between truck
customers. This is a search restriction, not a proven dominance result;
max_station_stops allows auditing longer paths. Stations may recur on later arcs.
The dynamic program retains a Pareto frontier in elapsed time and remaining
energy.
"""
function repair_fixed_bevtspd_route(
    instance::BEVTSPDInstance,
    truck_route::AbstractVector{Int},
    drone_route::AbstractVector{Int};
    base_objective::Real = NaN,
    max_station_stops::Int = min(1, length(instance.station_nodes)),
    station_path_mode::Symbol = :bounded,
    station_graph::Union{Nothing,StationParetoGraph} = nothing,
)
    _validate_instance_route(instance, truck_route, drone_route)
    isempty(intersect(Set(truck_route), Set(instance.station_nodes))) ||
        throw(ArgumentError("fixed route repair expects a station-free truck route"))
    route_energy_actions(
        truck_route,
        drone_route,
        instance.truck_energy_matrix,
        instance.drone_energy_matrix,
    )

    core_route = collect(truck_route)
    fixed_drone_route = collect(drone_route)
    drone_nodes = Set(fixed_drone_route)
    shared = [node for node in core_route if node in drone_nodes]
    truck_positions = Dict(node => index for (index, node) in enumerate(core_route))
    drone_positions = Dict(node => index for (index, node) in enumerate(fixed_drone_route))
    required_pairs = zip(core_route[1:end-1], core_route[2:end])
    station_options = if station_path_mode == :pareto
        _build_station_pareto_options(instance;
            required_pairs=required_pairs,graph=station_graph)
    elseif station_path_mode == :bounded
        _build_station_path_options(instance;
            required_pairs=required_pairs,
            max_station_stops=max_station_stops)
    else
        throw(ArgumentError("unsupported station_path_mode: $station_path_mode"))
    end
    labels = _RepairLabel[
        _RepairLabel(
            instance.energy_parameters.truck_capacity,
            0.0,
            [instance.start_depot],
        ),
    ]

    for shared_index in 1:(length(shared) - 1)
        launch = shared[shared_index]
        rendezvous = shared[shared_index + 1]
        truck_first = truck_positions[launch]
        truck_last = truck_positions[rendezvous]
        drone_first = drone_positions[launch]
        drone_last = drone_positions[rendezvous]
        drone_customer = drone_last - drone_first == 2 ?
            fixed_drone_route[drone_first + 1] : nothing
        expanded = _RepairLabel[]

        for label in labels
            entry_energy = label.energy
            drone_time = 0.0
            if !isnothing(drone_customer)
                sortie_energy =
                    instance.drone_energy_matrix[launch, drone_customer] +
                    instance.drone_energy_matrix[drone_customer, rendezvous]
                sortie_energy <= instance.energy_parameters.drone_capacity + 1e-9 ||
                    continue
                shared_debit =
                    instance.energy_parameters.shared_energy_factor * sortie_energy
                entry_energy + 1e-9 >= shared_debit || continue
                entry_energy = max(0.0, entry_energy - shared_debit)
                launch_service = launch == instance.start_depot ?
                    0.0 : instance.launch_time
                drone_time =
                    launch_service +
                    instance.drone_time_matrix[launch, drone_customer] +
                    instance.drone_time_matrix[drone_customer, rendezvous] +
                    instance.retrieve_time
            end

            interval_labels = _truck_interval_labels(
                core_route,
                truck_first,
                truck_last,
                entry_energy,
                station_options,
            )
            for interval in interval_labels
                push!(expanded, _RepairLabel(
                    interval.energy,
                    label.objective + max(interval.time, drone_time),
                    vcat(label.truck_route, interval.nodes[2:end]),
                ))
            end
        end

        isempty(expanded) && return EnergyICPResult(
            false,
            Inf,
            Int[],
            fixed_drone_route,
            NaN,
            Float64(base_objective),
        )
        labels = _prune_repair_labels(expanded)
    end

    best = argmin(label -> label.objective, labels)
    verified = evaluate_bevtspd_route(
        instance,
        best.truck_route,
        fixed_drone_route,
    )
    verified.feasible || error("station repair returned an infeasible route")
    isapprox(verified.objective, best.objective; atol = 1e-8) ||
        error("station repair objective disagrees with independent verifier")
    return EnergyICPResult(
        true,
        verified.objective,
        best.truck_route,
        fixed_drone_route,
        verified.simulation.final_energy,
        Float64(base_objective),
    )
end

function _drone_time_capacity(instance::BEVTSPDInstance)
    for index in eachindex(instance.drone_energy_matrix)
        energy = instance.drone_energy_matrix[index]
        if energy > 0
            return instance.energy_parameters.drone_capacity *
                instance.drone_time_matrix[index] / energy
        end
    end
    return Inf
end

"""
Energy-ICP baseline: run upstream ICP without station nodes, then solve the
fixed-route station insertion problem exactly.
"""
function solve_energy_icp(
    instance::BEVTSPDInstance;
    max_nodes::Int = 20,
    chain_initialization_method::Symbol = :FI,
    chainlet_initialization_method::Symbol = :FI,
    chainlet_solving_method::Symbol = :TSP_EP_all,
    search_method::Symbol = :Greedy,
    station_path_mode::Symbol = :bounded,
)
    local_nodes = vcat(instance.start_depot, instance.customer_nodes)
    truck_time = Matrix{Float64}(instance.truck_time_matrix[local_nodes, local_nodes])
    drone_time = Matrix{Float64}(instance.drone_time_matrix[local_nodes, local_nodes])
    base = solve_tspd(
        truck_time,
        drone_time;
        flying_range = _drone_time_capacity(instance),
        max_nodes = max_nodes,
        chain_initialization_method = chain_initialization_method,
        chainlet_initialization_method = chainlet_initialization_method,
        chainlet_solving_method = chainlet_solving_method,
        chainlet_evaluation_method = :Default,
        search_method = search_method,
    )

    local_dummy = length(local_nodes) + 1
    map_node(node::Int) = node == local_dummy ?
        instance.end_depot : local_nodes[node]
    truck_route = map(map_node, base.truck_route)
    drone_route = map(map_node, base.drone_route)
    return repair_fixed_bevtspd_route(
        instance,
        truck_route,
        drone_route;
        base_objective = base.total_cost,
        station_path_mode = station_path_mode,
    )
end

"""
Jointly reconsider drone-only customers and charging paths.

Each move returns one drone customer to the truck, tries every position in the
station-free truck order, and exactly reoptimizes station paths. Best-improvement
descent repeats until no role-return move improves synchronized completion time.
This is the first deterministic cross-layer neighborhood beyond Energy-ICP.
"""
function improve_drone_to_truck_roles(
    instance::BEVTSPDInstance,
    truck_route::AbstractVector{Int},
    drone_route::AbstractVector{Int};
    atol::Float64 = 1e-10,
)
    station_set = Set(instance.station_nodes)
    core_truck_route = [node for node in truck_route if !(node in station_set)]
    current_drone_route = collect(drone_route)
    current = repair_fixed_bevtspd_route(
        instance,
        core_truck_route,
        current_drone_route,
    )
    current.feasible || return EnergyRoleSearchResult(
        false,
        Inf,
        Int[],
        current_drone_route,
        NaN,
        0,
        0,
    )

    iterations = 0
    candidates_evaluated = 0
    while true
        drone_only_customers = [
            node for node in current_drone_route
            if node in instance.customer_nodes && !(node in core_truck_route)
        ]
        best_result = current
        best_core_route = core_truck_route
        best_drone_route = current_drone_route

        for customer in drone_only_customers
            candidate_drone_route = [node for node in current_drone_route if node != customer]
            for insertion_index in 1:(length(core_truck_route) - 1)
                candidate_core_route = copy(core_truck_route)
                insert!(candidate_core_route, insertion_index + 1, customer)
                candidate = repair_fixed_bevtspd_route(
                    instance,
                    candidate_core_route,
                    candidate_drone_route,
                )
                candidates_evaluated += 1
                if candidate.feasible && candidate.objective < best_result.objective - atol
                    best_result = candidate
                    best_core_route = candidate_core_route
                    best_drone_route = candidate_drone_route
                end
            end
        end

        best_result.objective < current.objective - atol || break
        current = best_result
        core_truck_route = best_core_route
        current_drone_route = best_drone_route
        iterations += 1
    end

    return EnergyRoleSearchResult(
        true,
        current.objective,
        current.truck_route,
        current.drone_route,
        current.final_energy,
        iterations,
        candidates_evaluated,
    )
end

"""Best-improvement exchange of one combined truck node and one drone node."""
function improve_truck_drone_role_exchange(
    instance::BEVTSPDInstance,
    truck_route::AbstractVector{Int},
    drone_route::AbstractVector{Int};
    atol::Float64 = 1e-10,
)
    station_set = Set(instance.station_nodes)
    core_truck_route = [node for node in truck_route if !(node in station_set)]
    current_drone_route = collect(drone_route)
    current = repair_fixed_bevtspd_route(instance, core_truck_route, current_drone_route)
    current.feasible || return EnergyRoleSearchResult(false, Inf, Int[], current_drone_route, NaN, 0, 0)

    iterations = 0
    candidates_evaluated = 0
    while true
        drone_only = [
            node for node in current_drone_route
            if node in instance.customer_nodes && !(node in core_truck_route)
        ]
        combined_customers = [
            node for node in core_truck_route
            if node in instance.customer_nodes && node in current_drone_route
        ]
        best_result = current
        best_core_route = core_truck_route
        best_drone_route = current_drone_route

        for truck_customer in combined_customers, drone_customer in drone_only
            swap_role(node) = node == truck_customer ? drone_customer :
                node == drone_customer ? truck_customer : node
            candidate_core_route = map(swap_role, core_truck_route)
            candidate_drone_route = map(swap_role, current_drone_route)
            candidate = repair_fixed_bevtspd_route(
                instance,
                candidate_core_route,
                candidate_drone_route,
            )
            candidates_evaluated += 1
            if candidate.feasible && candidate.objective < best_result.objective - atol
                best_result = candidate
                best_core_route = candidate_core_route
                best_drone_route = candidate_drone_route
            end
        end

        best_result.objective < current.objective - atol || break
        current = best_result
        core_truck_route = best_core_route
        current_drone_route = best_drone_route
        iterations += 1
    end

    return EnergyRoleSearchResult(
        true,
        current.objective,
        current.truck_route,
        current.drone_route,
        current.final_energy,
        iterations,
        candidates_evaluated,
    )
end

function improve_truck_drone_role_exchange(
    instance::BEVTSPDInstance,
    result::Union{EnergyICPResult, EnergyRoleSearchResult};
    kwargs...,
)
    return improve_truck_drone_role_exchange(
        instance,
        result.truck_route,
        result.drone_route;
        kwargs...,
    )
end

"""Relocate and 2-opt descent for an all-truck customer order."""
function improve_all_truck_order(
    instance::BEVTSPDInstance,
    truck_route::AbstractVector{Int};
    atol::Float64 = 1e-10,
)
    station_set = Set(instance.station_nodes)
    core_route = [node for node in truck_route if !(node in station_set)]
    current = repair_fixed_bevtspd_route(instance, core_route, core_route)
    current.feasible || return EnergyRoleSearchResult(false, Inf, Int[], Int[], NaN, 0, 0)
    iterations = 0
    candidates_evaluated = 0

    while true
        best_result = current
        best_core_route = core_route
        customer_positions = 2:(length(core_route) - 1)

        for first_index in customer_positions, last_index in first_index:last(customer_positions)
            first_index == last_index && continue
            candidate_core_route = copy(core_route)
            reverse!(candidate_core_route, first_index, last_index)
            candidate = repair_fixed_bevtspd_route(
                instance,
                candidate_core_route,
                candidate_core_route,
            )
            candidates_evaluated += 1
            if candidate.feasible && candidate.objective < best_result.objective - atol
                best_result = candidate
                best_core_route = candidate_core_route
            end
        end

        for from_index in customer_positions, after_index in 1:(length(core_route) - 1)
            after_index in (from_index - 1, from_index) && continue
            candidate_core_route = copy(core_route)
            customer = splice!(candidate_core_route, from_index)
            adjusted_after = after_index > from_index ? after_index - 1 : after_index
            insert!(candidate_core_route, adjusted_after + 1, customer)
            candidate = repair_fixed_bevtspd_route(
                instance,
                candidate_core_route,
                candidate_core_route,
            )
            candidates_evaluated += 1
            if candidate.feasible && candidate.objective < best_result.objective - atol
                best_result = candidate
                best_core_route = candidate_core_route
            end
        end

        best_result.objective < current.objective - atol || break
        current = best_result
        core_route = best_core_route
        iterations += 1
    end

    return EnergyRoleSearchResult(
        true,
        current.objective,
        current.truck_route,
        current.drone_route,
        current.final_energy,
        iterations,
        candidates_evaluated,
    )
end

"""Optimize sortie boundaries and charging paths for a fixed truck order/role split."""
function optimize_fixed_truck_order(
    instance::BEVTSPDInstance,
    truck_route::AbstractVector{Int},
    drone_customers::AbstractVector{Int};
    atol::Float64 = 1e-10,
)
    station_set = Set(instance.station_nodes)
    core_route = [node for node in truck_route if !(node in station_set)]
    first(core_route) == instance.start_depot ||
        throw(ArgumentError("truck order must start at the depot"))
    last(core_route) == instance.end_depot ||
        throw(ArgumentError("truck order must end at the dummy depot"))
    length(unique(core_route)) == length(core_route) ||
        throw(ArgumentError("truck order must contain unique core nodes"))
    length(unique(drone_customers)) == length(drone_customers) ||
        throw(ArgumentError("drone customer assignment must be unique"))
    isempty(intersect(Set(core_route), Set(drone_customers))) ||
        throw(ArgumentError("truck and drone customer assignments must be disjoint"))
    assigned = union(
        intersect(Set(core_route), Set(instance.customer_nodes)),
        Set(drone_customers),
    )
    assigned == Set(instance.customer_nodes) ||
        throw(ArgumentError("truck and drone assignments must cover every customer"))

    drone_bits = Dict(customer => (1 << (index - 1)) for
        (index, customer) in enumerate(drone_customers))
    all_drone_mask = (1 << length(drone_customers)) - 1
    required_pairs = zip(core_route[1:end-1], core_route[2:end])
    station_options = _build_station_path_options(
        instance;
        required_pairs = required_pairs,
    )
    frontiers = Dict{Tuple{Int, Int}, Vector{Tuple{Float64, Float64}}}()
    best_objective = Inf
    best_truck_route = Int[]
    best_drone_route = Int[]
    best_energy = NaN
    candidates_evaluated = 0

    function search!(
        core_index::Int,
        served_mask::Int,
        energy::Float64,
        objective::Float64,
        expanded_truck_route::Vector{Int},
        drone_route::Vector{Int},
    )
        objective >= best_objective - atol && return
        state = (core_index, served_mask)
        _state_is_dominated!(frontiers, state, objective, energy; atol = atol) && return

        if core_index == length(core_route)
            served_mask == all_drone_mask || return
            best_objective = objective
            best_truck_route = copy(expanded_truck_route)
            best_drone_route = copy(drone_route)
            best_energy = energy
            return
        end

        remaining_drone_count = count_ones(all_drone_mask & ~served_mask)
        remaining_core_arcs = length(core_route) - core_index
        remaining_drone_count <= remaining_core_arcs || return
        choices = Union{Nothing, Int}[nothing]
        append!(choices, [
            customer for customer in drone_customers
            if served_mask & drone_bits[customer] == 0
        ])

        for drone_customer in choices
            if isnothing(drone_customer)
                rendezvous_indices = (core_index + 1):(core_index + 1)
            else
                rendezvous_indices = (core_index + 1):length(core_route)
            end
            next_served_mask = isnothing(drone_customer) ?
                served_mask : served_mask | drone_bits[drone_customer]

            for rendezvous_index in rendezvous_indices
                entry_energy = energy
                drone_time = 0.0
                if !isnothing(drone_customer)
                    launch = core_route[core_index]
                    rendezvous = core_route[rendezvous_index]
                    sortie_energy =
                        instance.drone_energy_matrix[launch, drone_customer] +
                        instance.drone_energy_matrix[drone_customer, rendezvous]
                    sortie_energy <= instance.energy_parameters.drone_capacity + 1e-9 ||
                        continue
                    shared_debit =
                        instance.energy_parameters.shared_energy_factor * sortie_energy
                    energy + 1e-9 >= shared_debit || continue
                    entry_energy = max(0.0, energy - shared_debit)
                    launch_service = launch == instance.start_depot ?
                        0.0 : instance.launch_time
                    drone_time =
                        launch_service +
                        instance.drone_time_matrix[launch, drone_customer] +
                        instance.drone_time_matrix[drone_customer, rendezvous] +
                        instance.retrieve_time
                end

                interval_labels = _truck_interval_labels(
                    core_route,
                    core_index,
                    rendezvous_index,
                    entry_energy,
                    station_options,
                )
                for interval in interval_labels
                    candidates_evaluated += 1
                    next_objective = objective + max(interval.time, drone_time)
                    next_objective >= best_objective - atol && continue
                    next_truck_route = vcat(
                        expanded_truck_route,
                        interval.nodes[2:end],
                    )
                    next_drone_route = copy(drone_route)
                    !isnothing(drone_customer) && push!(
                        next_drone_route,
                        drone_customer,
                    )
                    push!(next_drone_route, core_route[rendezvous_index])
                    search!(
                        rendezvous_index,
                        next_served_mask,
                        interval.energy,
                        next_objective,
                        next_truck_route,
                        next_drone_route,
                    )
                end
            end
        end
    end

    search!(
        1,
        0,
        instance.energy_parameters.truck_capacity,
        0.0,
        [instance.start_depot],
        [instance.start_depot],
    )
    isfinite(best_objective) || return EnergyRoleSearchResult(
        false,
        Inf,
        Int[],
        Int[],
        NaN,
        0,
        candidates_evaluated,
    )

    verified = evaluate_bevtspd_route(instance, best_truck_route, best_drone_route)
    verified.feasible || error("fixed-order optimizer returned an infeasible route")
    isapprox(verified.objective, best_objective; atol = 1e-8) ||
        error("fixed-order optimizer objective disagrees with verifier")
    return EnergyRoleSearchResult(
        true,
        best_objective,
        best_truck_route,
        best_drone_route,
        best_energy,
        0,
        candidates_evaluated,
    )
end

"""Outer 2-opt/relocate search with nested sortie-boundary and energy DP."""
function improve_truck_order_with_sync_dp(
    instance::BEVTSPDInstance,
    truck_route::AbstractVector{Int},
    drone_route::AbstractVector{Int};
    atol::Float64 = 1e-10,
)
    station_set = Set(instance.station_nodes)
    core_route = [node for node in truck_route if !(node in station_set)]
    drone_customers = [
        node for node in drone_route
        if node in instance.customer_nodes && !(node in core_route)
    ]
    current = optimize_fixed_truck_order(
        instance,
        core_route,
        drone_customers;
        atol = atol,
    )
    current.feasible || return current
    iterations = 0
    candidates_evaluated = current.candidates_evaluated

    while true
        best = current
        best_core_route = core_route
        customer_positions = 2:(length(core_route) - 1)

        for first_index in customer_positions, last_index in first_index:last(customer_positions)
            first_index == last_index && continue
            candidate_core_route = copy(core_route)
            reverse!(candidate_core_route, first_index, last_index)
            candidate = optimize_fixed_truck_order(
                instance,
                candidate_core_route,
                drone_customers;
                atol = atol,
            )
            candidates_evaluated += candidate.candidates_evaluated
            if candidate.feasible && candidate.objective < best.objective - atol
                best = candidate
                best_core_route = candidate_core_route
            end
        end

        for from_index in customer_positions, after_index in 1:(length(core_route) - 1)
            after_index in (from_index - 1, from_index) && continue
            candidate_core_route = copy(core_route)
            customer = splice!(candidate_core_route, from_index)
            adjusted_after = after_index > from_index ? after_index - 1 : after_index
            insert!(candidate_core_route, adjusted_after + 1, customer)
            candidate = optimize_fixed_truck_order(
                instance,
                candidate_core_route,
                drone_customers;
                atol = atol,
            )
            candidates_evaluated += candidate.candidates_evaluated
            if candidate.feasible && candidate.objective < best.objective - atol
                best = candidate
                best_core_route = candidate_core_route
            end
        end

        best.objective < current.objective - atol || break
        current = best
        core_route = best_core_route
        iterations += 1
    end

    return EnergyRoleSearchResult(
        true,
        current.objective,
        current.truck_route,
        current.drone_route,
        current.final_energy,
        iterations,
        candidates_evaluated,
    )
end

"""Two-level look-ahead: role exchange followed by order/synchronization search."""
function improve_role_exchange_with_order_search(
    instance::BEVTSPDInstance,
    truck_route::AbstractVector{Int},
    drone_route::AbstractVector{Int};
    atol::Float64 = 1e-10,
)
    station_set = Set(instance.station_nodes)
    core_route = [node for node in truck_route if !(node in station_set)]
    current_drone_route = collect(drone_route)
    current = optimize_fixed_truck_order(
        instance,
        core_route,
        [
            node for node in current_drone_route
            if node in instance.customer_nodes && !(node in core_route)
        ];
        atol = atol,
    )
    current.feasible || return current
    total_candidates = current.candidates_evaluated
    iterations = 0

    while true
        drone_only = [
            node for node in current.drone_route
            if node in instance.customer_nodes && !(node in core_route)
        ]
        combined_customers = [
            node for node in core_route
            if node in instance.customer_nodes && node in current.drone_route
        ]
        best = current
        best_core_route = core_route

        for truck_customer in combined_customers, drone_customer in drone_only
            swap_role(node) = node == truck_customer ? drone_customer :
                node == drone_customer ? truck_customer : node
            candidate_core_route = map(swap_role, core_route)
            candidate_drone_route = map(swap_role, current.drone_route)
            candidate = improve_truck_order_with_sync_dp(
                instance,
                candidate_core_route,
                candidate_drone_route;
                atol = atol,
            )
            total_candidates += candidate.candidates_evaluated
            if candidate.feasible && candidate.objective < best.objective - atol
                best = candidate
                best_core_route = [
                    node for node in candidate.truck_route if !(node in station_set)
                ]
            end
        end

        best.objective < current.objective - atol || break
        current = best
        core_route = best_core_route
        iterations += 1
    end

    return EnergyRoleSearchResult(
        true,
        current.objective,
        current.truck_route,
        current.drone_route,
        current.final_energy,
        iterations,
        total_candidates,
    )
end

"""
Composite deterministic neighborhood: role return, role exchange, and a
look-ahead all-truck order search when exactly one drone customer remains.
"""
function improve_coupled_energy_neighborhoods(
    instance::BEVTSPDInstance,
    result::EnergyICPResult;
    atol::Float64 = 1e-10,
)
    current = EnergyRoleSearchResult(
        result.feasible,
        result.objective,
        result.truck_route,
        result.drone_route,
        result.final_energy,
        0,
        0,
    )
    current.feasible || return current
    total_iterations = 0
    total_candidates = 0

    while true
        role_return = improve_drone_to_truck_roles(
            instance,
            current.truck_route,
            current.drone_route;
            atol = atol,
        )
        role_exchange = improve_truck_drone_role_exchange(
            instance,
            current;
            atol = atol,
        )
        order_sync = improve_truck_order_with_sync_dp(
            instance,
            current.truck_route,
            current.drone_route;
            atol = atol,
        )
        role_order = improve_role_exchange_with_order_search(
            instance,
            current.truck_route,
            current.drone_route;
            atol = atol,
        )
        total_candidates += role_return.candidates_evaluated +
            role_exchange.candidates_evaluated + order_sync.candidates_evaluated +
            role_order.candidates_evaluated
        best = role_return.objective < role_exchange.objective ? role_return : role_exchange
        order_sync.objective < best.objective - atol && (best = order_sync)
        role_order.objective < best.objective - atol && (best = role_order)

        station_set = Set(instance.station_nodes)
        core_route = [node for node in current.truck_route if !(node in station_set)]
        drone_only = [
            node for node in current.drone_route
            if node in instance.customer_nodes && !(node in core_route)
        ]
        if length(drone_only) == 1
            customer = only(drone_only)
            for insertion_index in 1:(length(core_route) - 1)
                all_truck_route = copy(core_route)
                insert!(all_truck_route, insertion_index + 1, customer)
                order_search = improve_all_truck_order(
                    instance,
                    all_truck_route;
                    atol = atol,
                )
                total_candidates += order_search.candidates_evaluated
                order_search.objective < best.objective - atol && (best = order_search)
            end
        end

        best.objective < current.objective - atol || break
        current = best
        total_iterations += 1
    end

    return EnergyRoleSearchResult(
        true,
        current.objective,
        current.truck_route,
        current.drone_route,
        current.final_energy,
        total_iterations,
        total_candidates,
    )
end

function improve_drone_to_truck_roles(
    instance::BEVTSPDInstance,
    result::EnergyICPResult;
    kwargs...,
)
    return improve_drone_to_truck_roles(
        instance,
        result.truck_route,
        result.drone_route;
        kwargs...,
    )
end
