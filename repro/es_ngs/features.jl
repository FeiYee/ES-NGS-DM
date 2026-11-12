# ES-NGS deployable pre-repair feature encoder.
# Shapes: X[C,32,24], A[C,33,33], action[C,32], global[C,16], mask[C,33].

using Statistics

const FEAT_M = 32
const FEAT_NODE = 24
const FEAT_ACT = 32
const FEAT_GLOB = 16
const FEAT_TOK = FEAT_M + 1

function _core_route(inst, route)
    stations = Set(inst.station_nodes)
    return [node for node in route if !(node in stations)]
end

function _node_xy(inst, node)
    return (Float64(inst.coordinates[node, 1]), Float64(inst.coordinates[node, 2]))
end

function _store_min!(values::Dict{Int,Float64}, node::Int, value::Float64)
    values[node] = haskey(values, node) ? min(values[node], value) : value
end

"""State-of-charge values available from the current feasible solution."""
function _energy_state(inst, truck_route, drone_route)
    capacity = inst.energy_parameters.truck_capacity
    actions = route_energy_actions(
        truck_route,
        drone_route,
        inst.truck_energy_matrix,
        inst.drone_energy_matrix;
        swap_stations=Set(inst.station_nodes),
        swap_service_time=inst.swap_service_time,
    )
    simulation = simulate_energy(actions, capacity, inst.energy_parameters)
    simulation.feasible || error("feature encoder received an energy-infeasible incumbent")
    arrive = Dict{Int,Float64}(inst.start_depot => capacity)
    depart = Dict{Int,Float64}(inst.start_depot => capacity)
    for (index, action) in enumerate(actions)
        before = simulation.energy_trace[index]
        after = simulation.energy_trace[index + 1]
        if action isa TruckDrive
            _store_min!(depart, action.from, before)
            _store_min!(arrive, action.to, after)
        elseif action isa DroneSortie
            _store_min!(depart, action.launch, after)
            _store_min!(arrive, action.customer, after)
            _store_min!(depart, action.customer, after)
        elseif action isa FullTruckSwap
            _store_min!(arrive, action.station, before)
            _store_min!(depart, action.station, after)
        end
    end
    return (
        arrive=arrive,
        depart=depart,
        minimum=minimum(simulation.energy_trace),
        final=simulation.final_energy,
    )
end

"""Current/raw-route synchronization summaries; no repair result is used."""
function _sync_state(inst, truck_route, drone_route)
    station_set = Set(inst.station_nodes)
    drone_set = Set(drone_route)
    shared = [node for node in truck_route if node in drone_set]
    truck_position = Dict(node => index for (index, node) in enumerate(truck_route))
    drone_position = Dict(node => index for (index, node) in enumerate(drone_route))
    node_values = Dict{Int,NTuple{4,Float64}}()
    launch_nodes = Set{Int}()
    rendezvous_nodes = Set{Int}()
    total = 0.0
    truck_wait = 0.0
    drone_wait = 0.0
    sorties = 0
    drone_energy = 0.0
    high_energy_sorties = 0
    for shared_index in 1:max(0, length(shared) - 1)
        launch = shared[shared_index]
        rendezvous = shared[shared_index + 1]
        truck_first = truck_position[launch]
        truck_last = truck_position[rendezvous]
        drone_first = drone_position[launch]
        drone_last = drone_position[rendezvous]
        truck_time = 0.0
        for index in truck_first:(truck_last - 1)
            from = truck_route[index]
            to = truck_route[index + 1]
            truck_time += inst.truck_time_matrix[from, to]
            to in station_set && (truck_time += inst.swap_service_time)
        end
        drone_time = 0.0
        if drone_last - drone_first == 2
            customer = drone_route[drone_first + 1]
            energy = inst.drone_energy_matrix[launch, customer] +
                     inst.drone_energy_matrix[customer, rendezvous]
            drone_time = (launch == inst.start_depot ? 0.0 : inst.launch_time) +
                         inst.drone_time_matrix[launch, customer] +
                         inst.drone_time_matrix[customer, rendezvous] +
                         inst.retrieve_time
            sorties += 1
            drone_energy += energy
            energy >= 0.8 * inst.energy_parameters.drone_capacity && (high_energy_sorties += 1)
            push!(launch_nodes, launch)
            push!(rendezvous_nodes, rendezvous)
        end
        truck_wait_here = max(drone_time - truck_time, 0.0)
        drone_wait_here = max(truck_time - drone_time, 0.0)
        values = (truck_time, drone_time, truck_wait_here, drone_wait_here)
        for node in truck_route[truck_first:truck_last]
            node_values[node] = values
        end
        for node in drone_route[drone_first:drone_last]
            node_values[node] = values
        end
        total += max(truck_time, drone_time)
        truck_wait += truck_wait_here
        drone_wait += drone_wait_here
    end
    return (
        node_values=node_values,
        launch_nodes=launch_nodes,
        rendezvous_nodes=rendezvous_nodes,
        total=total,
        truck_wait=truck_wait,
        drone_wait=drone_wait,
        sorties=sorties,
        drone_energy=drone_energy,
        high_energy_sorties=high_energy_sorties,
    )
end

function _state_context(inst, before_tr, before_dr, before_T)
    core = _core_route(inst, before_tr)
    return (
        core=core,
        core_set=Set(core),
        truck_set=Set(before_tr),
        drone_set=Set(before_dr),
        station_set=Set(inst.station_nodes),
        energy=_energy_state(inst, before_tr, before_dr),
        sync=_sync_state(inst, before_tr, before_dr),
        before_T=Float64(before_T),
    )
end

function _select_local_nodes(inst, before_tr, before_dr, prop, context; M::Int=FEAT_M)
    new_core, new_drone, _, _, block_first, block_last = prop
    before_core = context.core
    first_index = clamp(block_first, 1, length(before_core))
    last_index = clamp(block_last, first_index, length(before_core))
    priority = Int[]
    append!(priority, before_core[first_index:last_index])
    append!(priority, before_core[max(1, first_index - 2):min(length(before_core), last_index + 2)])
    append!(priority, new_drone)
    append!(priority, before_dr)
    center = before_core[clamp(div(first_index + last_index, 2), 1, length(before_core))]
    stations = sort(
        collect(inst.station_nodes);
        by=station -> inst.truck_energy_matrix[center, station],
    )
    append!(priority, stations)
    append!(priority, new_core)
    append!(priority, before_core)
    push!(priority, inst.start_depot)
    push!(priority, inst.end_depot)
    seen = Set{Int}()
    selected = Int[]
    for node in priority
        node in seen && continue
        push!(seen, node)
        push!(selected, node)
        length(selected) >= M && break
    end
    valid_count = length(selected)
    pad_node = isempty(selected) ? inst.start_depot : selected[end]
    while length(selected) < M
        push!(selected, pad_node)
    end
    return selected[1:M], min(valid_count, M)
end

function _role_onehot(inst, node, context)
    if node == inst.start_depot || node == inst.end_depot
        return (1.0, 0.0, 0.0, 0.0)
    elseif node in context.station_set
        return (0.0, 1.0, 0.0, 0.0)
    elseif node in context.drone_set && !(node in context.truck_set)
        return (0.0, 0.0, 0.0, 1.0)
    else
        return (0.0, 0.0, 1.0, 0.0)
    end
end

function _route_time(inst, route)
    return sum(inst.truck_time_matrix[from, to] for (from, to) in zip(route, route[2:end]); init=0.0)
end

function _route_energy(inst, route)
    return sum(inst.truck_energy_matrix[from, to] for (from, to) in zip(route, route[2:end]); init=0.0)
end

function encode_candidate_features(
    inst,
    before_tr,
    before_dr,
    before_T,
    prop,
    budget_frac::Float64;
    M::Int=FEAT_M,
    context=nothing,
)
    isnothing(context) && (context = _state_context(inst, before_tr, before_dr, before_T))
    local_nodes, valid_count = _select_local_nodes(inst, before_tr, before_dr, prop, context; M=M)
    new_core, new_drone, removed, rebuilt, block_first, block_last = prop
    before_core = context.core
    new_core_set = Set(new_core)
    new_drone_set = Set(new_drone)
    index_by_node = Dict(node => index for (index, node) in enumerate(local_nodes[1:valid_count]))
    capacity = inst.energy_parameters.truck_capacity
    time_scale = max(context.before_T, 1e-6)

    X = zeros(Float64, M, FEAT_NODE)
    mask = zeros(Float64, FEAT_TOK)
    mask[1] = 1.0
    for i in 1:valid_count
        node = local_nodes[i]
        mask[i + 1] = 1.0
        x, y = _node_xy(inst, node)
        X[i, 1] = x / 20.0
        X[i, 2] = y / 20.0
        position = findfirst(==(node), before_core)
        if !isnothing(position) && position > 1
            X[i, 3] = inst.truck_time_matrix[before_core[position - 1], node]
        end
        if !isnothing(position) && position < length(before_core)
            X[i, 4] = inst.truck_time_matrix[node, before_core[position + 1]]
        end

        arrival = get(context.energy.arrive, node, capacity)
        departure = get(context.energy.depart, node, arrival)
        nearest_station = isempty(inst.station_nodes) ? capacity : minimum(
            inst.truck_energy_matrix[node, station] for station in inst.station_nodes
        )
        X[i, 5] = clamp(arrival / capacity, 0.0, 1.0)
        X[i, 6] = clamp(departure / capacity, 0.0, 1.0)
        X[i, 7] = clamp((departure - nearest_station) / capacity, -1.0, 1.0)
        X[i, 8] = nearest_station / capacity

        sync_values = get(context.sync.node_values, node, (0.0, 0.0, 0.0, 0.0))
        X[i, 9] = sync_values[1] / time_scale
        X[i, 10] = sync_values[2] / time_scale
        X[i, 11] = sync_values[3] / time_scale
        X[i, 12] = sync_values[4] / time_scale

        X[i, 13] = node in context.drone_set ? 1.0 : 0.0
        X[i, 14] = node in context.drone_set && !(node in context.truck_set) ? 1.0 : 0.0
        X[i, 15] = node in context.sync.launch_nodes ? 1.0 : 0.0
        X[i, 16] = node in context.sync.rendezvous_nodes ? 1.0 : 0.0

        role = _role_onehot(inst, node, context)
        X[i, 17] = role[1]
        X[i, 18] = role[2]
        X[i, 19] = role[3]
        X[i, 20] = role[4]
        affected = !isnothing(position) && block_first <= position <= block_last
        X[i, 21] = affected ? 1.0 : 0.0
        X[i, 22] = node in context.core_set && !(node in new_core_set) ? 1.0 : 0.0
        X[i, 23] = !(node in context.core_set) && node in new_core_set ? 1.0 : 0.0
        X[i, 24] = (!isnothing(position) && (position == block_first || position == block_last)) ? 1.0 : 0.0
    end

    # Relation ids: 0 none, 1 incumbent truck, 2 candidate truck,
    # 3 incumbent drone, 4 candidate drone, 5 synchronization pair,
    # 6 station proximity, 7 CLS.
    A = zeros(Int, FEAT_TOK, FEAT_TOK)
    for i in 1:FEAT_TOK, j in 1:FEAT_TOK
        if i == 1 || j == 1
            A[i, j] = 7
        end
    end
    function add_edges!(route, relation)
        for (from, to) in zip(route, route[2:end])
            if haskey(index_by_node, from) && haskey(index_by_node, to)
                fi = index_by_node[from] + 1
                ti = index_by_node[to] + 1
                A[fi, ti] = relation
                A[ti, fi] = relation
            end
        end
    end
    add_edges!(before_core, 1)
    add_edges!(new_core, 2)
    add_edges!(before_dr, 3)
    add_edges!(new_drone, 4)
    shared_nodes = [node for node in before_core if node in context.drone_set]
    add_edges!(shared_nodes, 5)
    for station in inst.station_nodes
        haskey(index_by_node, station) || continue
        nearest = nothing
        nearest_distance = Inf
        for node in local_nodes[1:valid_count]
            node == station && continue
            distance = inst.truck_energy_matrix[node, station]
            if distance < nearest_distance
                nearest = node
                nearest_distance = distance
            end
        end
        if !isnothing(nearest) && haskey(index_by_node, nearest)
            si = index_by_node[station] + 1
            ni = index_by_node[nearest] + 1
            A[si, ni] = 6
            A[ni, si] = 6
        end
    end

    candidate_sync = _sync_state(inst, new_core, new_drone)
    action = zeros(Float64, FEAT_ACT)
    action[1] = 1.0
    action[5] = (block_last - block_first + 1) / max(length(before_core), 1)
    action[6] = block_first / max(length(before_core), 1)
    action[7] = Float64(removed)
    action[8] = Float64(rebuilt)
    action[9] = _route_time(inst, before_core) - _route_time(inst, new_core)
    action[10] = length(new_drone) - length(before_dr)
    action[17] = _route_energy(inst, before_core) - _route_energy(inst, new_core)
    action[18] = context.energy.minimum / capacity
    action[19] = count(node -> node in context.station_set, before_tr) / max(length(before_tr), 1)
    action[20] = inst.energy_parameters.shared_energy_factor *
                 (context.sync.drone_energy - candidate_sync.drone_energy) / capacity
    action[25] = context.sync.total - candidate_sync.total
    action[26] = context.sync.truck_wait - candidate_sync.truck_wait
    action[27] = context.sync.drone_wait - candidate_sync.drone_wait
    action[28] = context.sync.sorties - candidate_sync.sorties

    global_feat = zeros(Float64, FEAT_GLOB)
    global_feat[1] = length(inst.customer_nodes) / 1000.0
    global_feat[2] = length(inst.station_nodes) / 20.0
    drone_only = count(node -> node in context.drone_set && !(node in context.truck_set), inst.customer_nodes)
    global_feat[3] = drone_only / max(length(inst.customer_nodes), 1)
    global_feat[4] = count(node -> node in context.station_set, before_tr) / 20.0
    global_feat[5] = (
        _route_energy(inst, before_tr) +
        inst.energy_parameters.shared_energy_factor * context.sync.drone_energy
    ) / max(capacity, 1.0)
    global_feat[6] = context.energy.minimum / capacity
    global_feat[7] = context.before_T / 40.0
    global_feat[8] = context.sync.truck_wait / time_scale
    global_feat[9] = context.sync.drone_wait / time_scale
    global_feat[10] = context.sync.sorties / max(length(inst.customer_nodes), 1)
    global_feat[11] = context.energy.final / capacity
    global_feat[12] = context.sync.high_energy_sorties / max(context.sync.sorties, 1)
    global_feat[16] = clamp(budget_frac, 0.0, 1.0)

    return (
        X=X,
        A=A,
        action=action,
        global_feat=global_feat,
        mask=mask,
        local_nodes=local_nodes,
    )
end

function encode_batch(inst, before_tr, before_dr, before_T, props, budget_frac::Float64)
    candidate_count = length(props)
    X = zeros(Float64, candidate_count, FEAT_M, FEAT_NODE)
    A = zeros(Int, candidate_count, FEAT_TOK, FEAT_TOK)
    action = zeros(Float64, candidate_count, FEAT_ACT)
    global_feat = zeros(Float64, candidate_count, FEAT_GLOB)
    mask = zeros(Float64, candidate_count, FEAT_TOK)
    context = _state_context(inst, before_tr, before_dr, before_T)
    for (candidate, proposal) in enumerate(props)
        features = encode_candidate_features(
            inst,
            before_tr,
            before_dr,
            before_T,
            proposal,
            budget_frac;
            context=context,
        )
        X[candidate, :, :] .= features.X
        A[candidate, :, :] .= features.A
        action[candidate, :] .= features.action
        global_feat[candidate, :] .= features.global_feat
        mask[candidate, :] .= features.mask
    end
    return (X=X, A=A, action=action, global_feat=global_feat, mask=mask)
end
