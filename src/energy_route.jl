"""
Convert an ICP truck/drone solution into shared-battery actions.

The terminal depot must use a distinct dummy-node id, as in the upstream ICP
representation. Customer and synchronization nodes must be unique, while a
physical swap-station id may occur multiple times in the truck route. A drone
debit is placed at launch, before the truck traverses the corresponding
synchronization segment. A full swap is placed immediately after the truck
arrives at a battery-swap station.
"""
function route_energy_actions(
    truck_route::AbstractVector{Int},
    drone_route::AbstractVector{Int},
    truck_energy_matrix::AbstractMatrix{<:Real},
    drone_energy_matrix::AbstractMatrix{<:Real};
    swap_stations::AbstractSet{Int} = Set{Int}(),
    swap_service_time::Real = 0.0,
    prohibit_station_rendezvous::Bool = true,
)
    length(truck_route) >= 2 || throw(ArgumentError("truck route must contain at least two nodes"))
    length(drone_route) >= 2 || throw(ArgumentError("drone route must contain at least two nodes"))
    truck_nonstations = [node for node in truck_route if !(node in swap_stations)]
    length(unique(truck_nonstations)) == length(truck_nonstations) || throw(ArgumentError(
        "truck customer/synchronization nodes must be unique; use a dummy terminal depot",
    ))
    length(unique(drone_route)) == length(drone_route) ||
        throw(ArgumentError("drone route nodes must be unique; use a dummy terminal depot"))

    truck_size = size(truck_energy_matrix)
    drone_size = size(drone_energy_matrix)
    length(truck_size) == 2 && truck_size[1] == truck_size[2] ||
        throw(DimensionMismatch("truck energy matrix must be square"))
    drone_size == truck_size ||
        throw(DimensionMismatch("truck and drone energy matrices must have equal size"))
    all(node -> 1 <= node <= truck_size[1], truck_route) ||
        throw(BoundsError(truck_energy_matrix, truck_route))
    all(node -> 1 <= node <= truck_size[1], drone_route) ||
        throw(BoundsError(drone_energy_matrix, drone_route))

    truck_nodes = Set(truck_route)
    drone_nodes = Set(drone_route)
    truck_shared = [node for node in truck_route if node in drone_nodes]
    drone_shared = [node for node in drone_route if node in truck_nodes]
    isempty(truck_shared) &&
        throw(ArgumentError("truck and drone routes must share synchronization nodes"))
    length(unique(truck_shared)) == length(truck_shared) ||
        throw(ArgumentError("synchronization nodes must be unique"))
    truck_shared == drone_shared ||
        throw(ArgumentError("truck and drone synchronization nodes must have the same order"))
    first(truck_route) == first(drone_route) == first(truck_shared) ||
        throw(ArgumentError("truck and drone routes must start at the same depot"))
    last(truck_route) == last(drone_route) == last(truck_shared) ||
        throw(ArgumentError("truck and drone routes must end at the same dummy depot"))

    if prohibit_station_rendezvous
        invalid_stations = intersect(swap_stations, Set(truck_shared))
        isempty(invalid_stations) || throw(ArgumentError(
            "swap stations cannot be drone launch/rendezvous nodes: $(sort!(collect(invalid_stations)))",
        ))
    end

    truck_positions = Dict(
        node => findfirst(==(node), truck_route) for node in truck_shared
    )
    drone_positions = Dict(node => index for (index, node) in enumerate(drone_route))
    actions = AbstractEnergyAction[]

    for shared_index in 1:(length(truck_shared) - 1)
        launch = truck_shared[shared_index]
        rendezvous = truck_shared[shared_index + 1]
        truck_first = truck_positions[launch]
        truck_last = truck_positions[rendezvous]
        drone_first = drone_positions[launch]
        drone_last = drone_positions[rendezvous]

        truck_first < truck_last ||
            throw(ArgumentError("truck synchronization order is not increasing"))
        drone_first < drone_last ||
            throw(ArgumentError("drone synchronization order is not increasing"))
        drone_segment_length = drone_last - drone_first + 1
        drone_segment_length in (2, 3) || throw(ArgumentError(
            "each drone sortie must serve at most one customer",
        ))

        if drone_segment_length == 3
            customer = drone_route[drone_first + 1]
            drone_energy =
                Float64(drone_energy_matrix[launch, customer]) +
                Float64(drone_energy_matrix[customer, rendezvous])
            drone_energy >= 0 || throw(ArgumentError("drone energy must be nonnegative"))
            push!(actions, DroneSortie(launch, customer, rendezvous, drone_energy))
        end

        for truck_index in truck_first:(truck_last - 1)
            from = truck_route[truck_index]
            to = truck_route[truck_index + 1]
            truck_energy = Float64(truck_energy_matrix[from, to])
            truck_energy >= 0 || throw(ArgumentError("truck energy must be nonnegative"))
            push!(actions, TruckDrive(from, to, truck_energy))
            if to in swap_stations
                push!(actions, FullTruckSwap(to, swap_service_time))
            end
        end
    end

    return actions
end

"""Build route actions and independently simulate their shared-energy state."""
function simulate_route_energy(
    truck_route::AbstractVector{Int},
    drone_route::AbstractVector{Int},
    truck_energy_matrix::AbstractMatrix{<:Real},
    drone_energy_matrix::AbstractMatrix{<:Real},
    parameters::EnergyParameters;
    initial_energy::Real = parameters.truck_capacity,
    kwargs...,
)
    actions = route_energy_actions(
        truck_route,
        drone_route,
        truck_energy_matrix,
        drone_energy_matrix;
        kwargs...,
    )
    simulation = simulate_energy(actions, initial_energy, parameters)
    return (actions = actions, simulation = simulation)
end
