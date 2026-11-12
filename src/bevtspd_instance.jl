using Random

"""Data for the shared-energy EVTSPD with fixed-time full charging."""
struct BEVTSPDInstance
    coordinates::Matrix{Float64}
    customer_nodes::Vector{Int}
    station_nodes::Vector{Int}
    start_depot::Int
    end_depot::Int
    truck_time_matrix::Matrix{Float64}
    drone_time_matrix::Matrix{Float64}
    truck_energy_matrix::Matrix{Float64}
    drone_energy_matrix::Matrix{Float64}
    energy_parameters::EnergyParameters
    swap_service_time::Float64
    launch_time::Float64
    retrieve_time::Float64
end

"""
Generate the random geometry used in the EVTSPD-FF numerical study.

Customers and stations are uniform in `[-coordinate_limit, coordinate_limit]^2`.
The original study used roughly one station per three customers and at least
three stations. Generation is deterministic for a fixed RNG/seed, but does not
by itself certify that a particular routing restriction is feasible.
"""
function generate_random_bevtspd_instance(
    rng::AbstractRNG,
    customer_count::Int;
    station_count::Int = max(3, cld(customer_count, 3)),
    coordinate_limit::Real = 20.0,
    kwargs...,
)
    customer_count >= 1 || throw(ArgumentError("customer count must be positive"))
    station_count >= 0 || throw(ArgumentError("station count must be nonnegative"))
    coordinate_limit > 0 || throw(ArgumentError("coordinate limit must be positive"))

    scale = 2.0 * Float64(coordinate_limit)
    offset = Float64(coordinate_limit)
    customers = scale .* rand(rng, customer_count, 2) .- offset
    stations = scale .* rand(rng, station_count, 2) .- offset
    return build_bevtspd_instance(customers, stations; kwargs...)
end

function generate_random_bevtspd_instance(
    seed::Integer,
    customer_count::Int;
    kwargs...,
)
    return generate_random_bevtspd_instance(
        MersenneTwister(seed),
        customer_count;
        kwargs...,
    )
end

function _coordinate_matrix(points, label::String)
    matrix = Matrix{Float64}(points)
    size(matrix, 2) == 2 ||
        throw(DimensionMismatch("$label coordinates must have two columns"))
    all(isfinite, matrix) || throw(ArgumentError("$label coordinates must be finite"))
    return matrix
end

"""
Build an EVTSPD-FF instance from customer and charging-station coordinates.

Defaults follow Zhu, Boyles, and Unnikrishnan's numerical setting. Time is in
hours, distance/energy is in kilometres, and the terminal depot is a distinct
dummy node colocated with the start depot.
"""
function build_bevtspd_instance(
    customer_coordinates,
    station_coordinates;
    depot::Tuple{<:Real, <:Real} = (0.0, 0.0),
    truck_speed::Real = 40.0,
    drone_speed::Real = 60.0,
    truck_capacity::Real = 100.0,
    drone_capacity::Real = 20.0,
    shared_energy_factor::Real = 0.4,
    swap_service_time::Real = 2.0 / 60.0,
    launch_time::Real = 100.0 / 3600.0,
    retrieve_time::Real = 20.0 / 3600.0,
)
    truck_speed > 0 || throw(ArgumentError("truck speed must be positive"))
    drone_speed > 0 || throw(ArgumentError("drone speed must be positive"))
    swap_service_time >= 0 || throw(ArgumentError("swap service time must be nonnegative"))
    launch_time >= 0 || throw(ArgumentError("launch time must be nonnegative"))
    retrieve_time >= 0 || throw(ArgumentError("retrieve time must be nonnegative"))

    customers = _coordinate_matrix(customer_coordinates, "customer")
    stations = _coordinate_matrix(station_coordinates, "station")
    depot_row = reshape(Float64[depot[1], depot[2]], 1, 2)
    coordinates = vcat(depot_row, customers, stations, depot_row)

    customer_count = size(customers, 1)
    station_count = size(stations, 1)
    customer_nodes = collect(2:(customer_count + 1))
    station_nodes = collect((customer_count + 2):(customer_count + station_count + 1))
    start_depot = 1
    end_depot = size(coordinates, 1)

    node_count = size(coordinates, 1)
    truck_distance = zeros(node_count, node_count)
    drone_distance = zeros(node_count, node_count)
    for from in 1:node_count, to in 1:node_count
        dx = abs(coordinates[from, 1] - coordinates[to, 1])
        dy = abs(coordinates[from, 2] - coordinates[to, 2])
        truck_distance[from, to] = dx + dy
        drone_distance[from, to] = hypot(dx, dy)
    end

    return BEVTSPDInstance(
        coordinates,
        customer_nodes,
        station_nodes,
        start_depot,
        end_depot,
        truck_distance ./ Float64(truck_speed),
        drone_distance ./ Float64(drone_speed),
        truck_distance,
        drone_distance,
        EnergyParameters(truck_capacity, drone_capacity, shared_energy_factor),
        Float64(swap_service_time),
        Float64(launch_time),
        Float64(retrieve_time),
    )
end

function _validate_instance_route(
    instance::BEVTSPDInstance,
    truck_route::AbstractVector{Int},
    drone_route::AbstractVector{Int},
)
    first(truck_route) == instance.start_depot &&
        first(drone_route) == instance.start_depot ||
        throw(ArgumentError("both routes must start at the instance depot"))
    last(truck_route) == instance.end_depot &&
        last(drone_route) == instance.end_depot ||
        throw(ArgumentError("both routes must end at the dummy depot"))

    station_set = Set(instance.station_nodes)
    isempty(intersect(station_set, Set(drone_route))) ||
        throw(ArgumentError("the drone route cannot visit charging stations"))
    visited_customers = intersect(
        Set(vcat(truck_route, drone_route)),
        Set(instance.customer_nodes),
    )
    visited_customers == Set(instance.customer_nodes) ||
        throw(ArgumentError("every customer must be visited"))

    allowed_nodes = Set(vcat(
        instance.start_depot,
        instance.customer_nodes,
        instance.station_nodes,
        instance.end_depot,
    ))
    all(node -> node in allowed_nodes, truck_route) ||
        throw(ArgumentError("truck route contains a node outside the instance"))
    all(node -> node in allowed_nodes, drone_route) ||
        throw(ArgumentError("drone route contains a node outside the instance"))
    return nothing
end

"""Evaluate synchronized operational time and exact shared-energy feasibility."""
function evaluate_bevtspd_route(
    instance::BEVTSPDInstance,
    truck_route::AbstractVector{Int},
    drone_route::AbstractVector{Int};
    initial_energy::Real = instance.energy_parameters.truck_capacity,
)
    _validate_instance_route(instance, truck_route, drone_route)
    station_set = Set(instance.station_nodes)
    energy_result = simulate_route_energy(
        truck_route,
        drone_route,
        instance.truck_energy_matrix,
        instance.drone_energy_matrix,
        instance.energy_parameters;
        initial_energy = initial_energy,
        swap_stations = station_set,
        swap_service_time = instance.swap_service_time,
    )

    truck_nodes = Set(truck_route)
    drone_nodes = Set(drone_route)
    shared = [node for node in truck_route if node in drone_nodes]
    truck_positions = Dict(node => index for (index, node) in enumerate(truck_route))
    drone_positions = Dict(node => index for (index, node) in enumerate(drone_route))
    operational_time = 0.0

    for shared_index in 1:(length(shared) - 1)
        launch = shared[shared_index]
        rendezvous = shared[shared_index + 1]
        truck_first = truck_positions[launch]
        truck_last = truck_positions[rendezvous]
        drone_first = drone_positions[launch]
        drone_last = drone_positions[rendezvous]

        truck_time = 0.0
        for index in truck_first:(truck_last - 1)
            from = truck_route[index]
            to = truck_route[index + 1]
            truck_time += instance.truck_time_matrix[from, to]
            to in station_set && (truck_time += instance.swap_service_time)
        end

        drone_time = 0.0
        if drone_last - drone_first == 2
            customer = drone_route[drone_first + 1]
            launch_service = launch == instance.start_depot ? 0.0 : instance.launch_time
            drone_time =
                launch_service +
                instance.drone_time_matrix[launch, customer] +
                instance.drone_time_matrix[customer, rendezvous] +
                instance.retrieve_time
        end
        operational_time += max(truck_time, drone_time)
    end

    feasible = energy_result.simulation.feasible
    return (
        feasible = feasible,
        objective = feasible ? operational_time : Inf,
        operational_time = operational_time,
        actions = energy_result.actions,
        simulation = energy_result.simulation,
    )
end
