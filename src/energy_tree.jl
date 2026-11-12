"""
Segment tree over exact energy-transfer maps.

The tree preserves route order when combining children, so the root is the
exact transfer map for the complete action sequence. Point updates and range
queries take logarithmic time.
"""
mutable struct EnergyTransferTree
    parameters::EnergyParameters
    action_count::Int
    leaf_base::Int
    nodes::Vector{EnergyTransfer}
    actions::Vector{AbstractEnergyAction}
end

function EnergyTransferTree(
    actions::AbstractVector{<:AbstractEnergyAction},
    parameters::EnergyParameters,
)
    action_count = length(actions)
    leaf_base = nextpow(2, max(1, action_count))
    nodes = fill(identity_energy_transfer(), 2 * leaf_base)
    stored_actions = AbstractEnergyAction[actions...]

    for index in eachindex(stored_actions)
        nodes[leaf_base + index - 1] =
            primitive_energy_transfer(stored_actions[index], parameters)
    end
    for index in (leaf_base - 1):-1:1
        nodes[index] = compose_energy_transfers(
            nodes[2 * index],
            nodes[2 * index + 1],
            parameters.truck_capacity,
        )
    end

    return EnergyTransferTree(
        parameters,
        action_count,
        leaf_base,
        nodes,
        stored_actions,
    )
end

full_energy_transfer(tree::EnergyTransferTree) = tree.nodes[1]

function update_energy_action!(
    tree::EnergyTransferTree,
    index::Int,
    action::AbstractEnergyAction,
)
    1 <= index <= tree.action_count || throw(BoundsError(tree.actions, index))
    tree.actions[index] = action

    node = tree.leaf_base + index - 1
    tree.nodes[node] = primitive_energy_transfer(action, tree.parameters)
    node = fld(node, 2)
    while node >= 1
        tree.nodes[node] = compose_energy_transfers(
            tree.nodes[2 * node],
            tree.nodes[2 * node + 1],
            tree.parameters.truck_capacity,
        )
        node == 1 && break
        node = fld(node, 2)
    end
    return tree
end

"""Return the exact transfer map for the inclusive action interval."""
function query_energy_transfer(tree::EnergyTransferTree, first::Int, last::Int)
    if first > last
        return identity_energy_transfer()
    end
    1 <= first <= last <= tree.action_count ||
        throw(BoundsError(tree.actions, first:last))

    left_node = tree.leaf_base + first - 1
    right_node = tree.leaf_base + last - 1
    left_accumulator = identity_energy_transfer()
    right_accumulator = identity_energy_transfer()

    while left_node <= right_node
        if isodd(left_node)
            left_accumulator = compose_energy_transfers(
                left_accumulator,
                tree.nodes[left_node],
                tree.parameters.truck_capacity,
            )
            left_node += 1
        end
        if iseven(right_node)
            right_accumulator = compose_energy_transfers(
                tree.nodes[right_node],
                right_accumulator,
                tree.parameters.truck_capacity,
            )
            right_node -= 1
        end
        left_node = fld(left_node, 2)
        right_node = fld(right_node, 2)
    end

    return compose_energy_transfers(
        left_accumulator,
        right_accumulator,
        tree.parameters.truck_capacity,
    )
end

"""
Evaluate a variable-length local rewrite without mutating the incumbent tree.

The replacement substitutes inclusive interval `first:last`. The returned
`suffix_required_input` is the minimum SoC that the replacement must preserve
at its exit to keep the incumbent suffix feasible.
"""
function evaluate_energy_replacement(
    tree::EnergyTransferTree,
    first::Int,
    last::Int,
    replacement::AbstractVector{<:AbstractEnergyAction};
    initial_energy::Real = tree.parameters.truck_capacity,
)
    1 <= first <= last <= tree.action_count ||
        throw(BoundsError(tree.actions, first:last))

    prefix = first == 1 ?
        identity_energy_transfer() : query_energy_transfer(tree, 1, first - 1)
    suffix = last == tree.action_count ?
        identity_energy_transfer() : query_energy_transfer(tree, last + 1, tree.action_count)
    replacement_transfer = summarize_energy(replacement, tree.parameters)

    candidate = compose_energy_transfers(
        compose_energy_transfers(
            prefix,
            replacement_transfer,
            tree.parameters.truck_capacity,
        ),
        suffix,
        tree.parameters.truck_capacity,
    )
    final_energy = evaluate_energy_transfer(candidate, initial_energy)

    return (
        feasible = !isnothing(final_energy),
        final_energy = final_energy,
        transfer = candidate,
        suffix_required_input = suffix.required_input,
    )
end
