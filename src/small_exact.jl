"""Result from exhaustive optimization of a small BEVTSPD instance."""
struct ExactBEVTSPDResult
    feasible::Bool
    objective::Float64
    truck_route::Vector{Int}
    drone_route::Vector{Int}
    final_energy::Float64
    incumbent_updates::Int
end

struct _StationPathOption
    nodes::Vector{Int}
    time::Float64
    transfer::EnergyTransfer
end

struct _TruckIntervalLabel
    energy::Float64
    time::Float64
    nodes::Vector{Int}
end

function _station_sequences(stations::Vector{Int}, max_length::Int)
    0 <= max_length <= length(stations) || throw(ArgumentError(
        "station path limit must be between zero and the station count",
    ))
    sequences = Vector{Vector{Int}}([Int[]])
    current = Int[]
    used = falses(length(stations))

    function extend!()
        length(current) == max_length && return
        for index in eachindex(stations)
            used[index] && continue
            used[index] = true
            push!(current, stations[index])
            push!(sequences, copy(current))
            extend!()
            pop!(current)
            used[index] = false
        end
    end

    extend!()
    return sequences
end

function _build_station_path_options(
    instance::BEVTSPDInstance;
    max_station_stops::Int = min(1, length(instance.station_nodes)),
    required_pairs = nothing,
)
    sequences = _station_sequences(instance.station_nodes, max_station_stops)
    core_nodes = vcat(
        instance.start_depot,
        instance.customer_nodes,
        instance.end_depot,
    )
    options = Dict{Tuple{Int, Int}, Vector{_StationPathOption}}()
    pairs = isnothing(required_pairs) ?
        [(from, to) for from in core_nodes for to in core_nodes if from != to] :
        unique(collect(required_pairs))

    for (from, to) in pairs
        from == to && throw(ArgumentError("station-path endpoints must differ"))
        pair_options = _StationPathOption[]
        for stations in sequences
            nodes = vcat(from, stations, to)
            actions = AbstractEnergyAction[]
            path_time = 0.0
            for index in 1:(length(nodes) - 1)
                origin = nodes[index]
                destination = nodes[index + 1]
                push!(actions, TruckDrive(
                    origin,
                    destination,
                    instance.truck_energy_matrix[origin, destination],
                ))
                path_time += instance.truck_time_matrix[origin, destination]
                if destination in instance.station_nodes
                    push!(actions, FullTruckSwap(destination, instance.swap_service_time))
                    path_time += instance.swap_service_time
                end
            end
            transfer = summarize_energy(actions, instance.energy_parameters)
            transfer.feasible && push!(
                pair_options,
                _StationPathOption(nodes, path_time, transfer),
            )
        end
        options[(from, to)] = pair_options
    end
    return options
end

function _prune_interval_labels(labels::Vector{_TruckIntervalLabel}; atol::Float64 = 1e-10)
    kept = _TruckIntervalLabel[]
    for candidate in sort(labels; by = label -> (label.time, -label.energy))
        dominated = any(
            incumbent.time <= candidate.time + atol &&
            incumbent.energy + atol >= candidate.energy for incumbent in kept
        )
        dominated && continue
        filter!(kept) do incumbent
            !(candidate.time <= incumbent.time + atol &&
              candidate.energy + atol >= incumbent.energy)
        end
        push!(kept, candidate)
    end
    return kept
end

function _truck_interval_labels(
    core_route::Vector{Int},
    first_index::Int,
    last_index::Int,
    entry_energy::Float64,
    station_options::Dict{Tuple{Int, Int}, Vector{_StationPathOption}},
)
    labels = _TruckIntervalLabel[
        _TruckIntervalLabel(entry_energy, 0.0, [core_route[first_index]]),
    ]
    for index in first_index:(last_index - 1)
        expanded = _TruckIntervalLabel[]
        for label in labels
            for option in station_options[(core_route[index], core_route[index + 1])]
                exit_energy = evaluate_energy_transfer(option.transfer, label.energy)
                isnothing(exit_energy) && continue
                push!(expanded, _TruckIntervalLabel(
                    exit_energy,
                    label.time + option.time,
                    vcat(label.nodes, option.nodes[2:end]),
                ))
            end
        end
        isempty(expanded) && return expanded
        labels = _prune_interval_labels(expanded)
    end
    return labels
end

function _visit_permutations!(visit::Function, values::Vector{Int}, first_index::Int)
    if first_index > length(values)
        visit(copy(values))
        return
    end
    for index in first_index:length(values)
        values[first_index], values[index] = values[index], values[first_index]
        _visit_permutations!(visit, values, first_index + 1)
        values[first_index], values[index] = values[index], values[first_index]
    end
end

function _state_is_dominated!(
    frontiers::Dict{Tuple{Int, Int}, Vector{Tuple{Float64, Float64}}},
    state::Tuple{Int, Int},
    cost::Float64,
    energy::Float64;
    atol::Float64 = 1e-10,
)
    frontier = get!(frontiers, state, Tuple{Float64, Float64}[])
    any(old_cost <= cost + atol && old_energy + atol >= energy for
        (old_cost, old_energy) in frontier) && return true
    filter!(frontier) do label
        old_cost, old_energy = label
        !(cost <= old_cost + atol && energy + atol >= old_energy)
    end
    push!(frontier, (cost, energy))
    return false
end

"""
Solve a small EVTSPD-FF instance by exhaustive enumeration with Pareto pruning.

The oracle enumerates truck/drone customer assignments, truck-customer orders,
synchronization intervals, and every simple charging-station path between two
consecutive truck customers. A station may be reused on different arcs. Under
the metric instance construction and fixed full-charge policy, repeating a
station within one customer-to-customer arc is dominated. The routine is meant
for correctness experiments, not production-sized optimization.
"""
function solve_small_bevtspd_exact(
    instance::BEVTSPDInstance;
    max_customers::Int = 7,
)
    customers = instance.customer_nodes
    customer_count = length(customers)
    customer_count <= max_customers || throw(ArgumentError(
        "exact oracle is limited to $max_customers customers; got $customer_count",
    ))
    customer_bits = Dict(customer => (1 << (index - 1)) for
        (index, customer) in enumerate(customers))
    all_customer_mask = (1 << customer_count) - 1
    station_options = _build_station_path_options(
        instance;
        max_station_stops = length(instance.station_nodes),
    )

    best_objective = Inf
    best_truck_route = Int[]
    best_drone_route = Int[]
    best_final_energy = NaN
    incumbent_updates = 0

    for drone_mask in 0:all_customer_mask
        truck_customers = [
            customer for customer in customers
            if drone_mask & customer_bits[customer] == 0
        ]
        _visit_permutations!(truck_customers, 1) do truck_order
            core_route = vcat(
                instance.start_depot,
                truck_order,
                instance.end_depot,
            )
            frontiers = Dict{Tuple{Int, Int}, Vector{Tuple{Float64, Float64}}}()

            function search!(
                core_index::Int,
                served_drone_mask::Int,
                energy::Float64,
                objective::Float64,
                truck_route::Vector{Int},
                drone_route::Vector{Int},
            )
                objective >= best_objective - 1e-10 && return
                state = (core_index, served_drone_mask)
                _state_is_dominated!(frontiers, state, objective, energy) && return

                if core_index == length(core_route)
                    served_drone_mask == drone_mask || return
                    incumbent_updates += 1
                    best_objective = objective
                    best_truck_route = copy(truck_route)
                    best_drone_route = copy(drone_route)
                    best_final_energy = energy
                    return
                end

                remaining_drone_count = count_ones(drone_mask & ~served_drone_mask)
                remaining_core_arcs = length(core_route) - core_index
                remaining_drone_count <= remaining_core_arcs || return

                # A no-sortie interval need not skip a truck customer: splitting
                # it preserves time/energy and leaves more rendezvous choices.
                choices = Union{Nothing, Int}[nothing]
                append!(choices, [
                    customer for customer in customers
                    if drone_mask & customer_bits[customer] != 0 &&
                       served_drone_mask & customer_bits[customer] == 0
                ])

                for drone_customer in choices
                    if isnothing(drone_customer)
                        rendezvous_indices = (core_index + 1):(core_index + 1)
                        entry_energy = energy
                        drone_time = 0.0
                        next_served_mask = served_drone_mask
                    else
                        rendezvous_indices = (core_index + 1):length(core_route)
                        entry_energy = energy
                        drone_time = 0.0
                        next_served_mask = served_drone_mask | customer_bits[drone_customer]
                    end

                    for rendezvous_index in rendezvous_indices
                        local_entry_energy = entry_energy
                        local_drone_time = drone_time
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
                            local_entry_energy = max(0.0, energy - shared_debit)
                            launch_service = launch == instance.start_depot ?
                                0.0 : instance.launch_time
                            local_drone_time =
                                launch_service +
                                instance.drone_time_matrix[launch, drone_customer] +
                                instance.drone_time_matrix[drone_customer, rendezvous] +
                                instance.retrieve_time
                        end

                        interval_labels = _truck_interval_labels(
                            core_route,
                            core_index,
                            rendezvous_index,
                            local_entry_energy,
                            station_options,
                        )
                        for label in interval_labels
                            ring_time = max(label.time, local_drone_time)
                            next_objective = objective + ring_time
                            next_objective >= best_objective - 1e-10 && continue
                            next_truck_route = vcat(truck_route, label.nodes[2:end])
                            next_drone_route = copy(drone_route)
                            !isnothing(drone_customer) && push!(
                                next_drone_route,
                                drone_customer,
                            )
                            push!(next_drone_route, core_route[rendezvous_index])
                            search!(
                                rendezvous_index,
                                next_served_mask,
                                label.energy,
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
        end
    end

    if isfinite(best_objective)
        verified = evaluate_bevtspd_route(
            instance,
            best_truck_route,
            best_drone_route,
        )
        verified.feasible || error("exact oracle returned an energy-infeasible route")
        isapprox(verified.objective, best_objective; atol = 1e-8) ||
            error("exact oracle objective disagrees with independent verifier")
        best_final_energy = verified.simulation.final_energy
    end

    return ExactBEVTSPDResult(
        isfinite(best_objective),
        best_objective,
        best_truck_route,
        best_drone_route,
        best_final_energy,
        incumbent_updates,
    )
end
