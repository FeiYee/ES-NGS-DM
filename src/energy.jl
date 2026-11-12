"""Actions that change the shared truck battery state."""
abstract type AbstractEnergyAction end

struct TruckDrive <: AbstractEnergyAction
    from::Int
    to::Int
    energy::Float64

    function TruckDrive(from::Int, to::Int, energy::Real)
        energy >= 0 || throw(ArgumentError("truck energy must be nonnegative"))
        new(from, to, Float64(energy))
    end
end

struct DroneSortie <: AbstractEnergyAction
    launch::Int
    customer::Int
    rendezvous::Int
    energy::Float64

    function DroneSortie(launch::Int, customer::Int, rendezvous::Int, energy::Real)
        energy >= 0 || throw(ArgumentError("drone energy must be nonnegative"))
        new(launch, customer, rendezvous, Float64(energy))
    end
end

struct FullTruckSwap <: AbstractEnergyAction
    station::Int
    service_time::Float64

    function FullTruckSwap(station::Int, service_time::Real = 0.0)
        service_time >= 0 || throw(ArgumentError("swap service time must be nonnegative"))
        new(station, Float64(service_time))
    end
end

struct EnergyParameters
    truck_capacity::Float64
    drone_capacity::Float64
    shared_energy_factor::Float64

    function EnergyParameters(
        truck_capacity::Real,
        drone_capacity::Real,
        shared_energy_factor::Real = 1.0,
    )
        truck_capacity > 0 || throw(ArgumentError("truck capacity must be positive"))
        drone_capacity > 0 || throw(ArgumentError("drone capacity must be positive"))
        shared_energy_factor >= 0 ||
            throw(ArgumentError("shared energy factor must be nonnegative"))
        new(
            Float64(truck_capacity),
            Float64(drone_capacity),
            Float64(shared_energy_factor),
        )
    end
end

"""
Exact shared-battery simulation result.

`energy_trace[k]` is the truck state of charge before action `k`; the final
entry is the state after the last action. A drone-capacity failure does not
change the truck state.
"""
struct EnergySimulationResult
    feasible::Bool
    final_energy::Float64
    energy_trace::Vector{Float64}
    violation_index::Union{Nothing, Int}
    violation::Union{Nothing, Symbol}
end

"""
Compact transfer map for a fixed segment.

For every feasible entry state `q >= required_input`, the segment exits with
`slope*q + intercept`. Full swaps make `slope == 0`; otherwise it is one.
"""
struct EnergyTransfer
    required_input::Float64
    slope::Int8
    intercept::Float64
    feasible::Bool
    reason::Symbol
end

identity_energy_transfer() = EnergyTransfer(0.0, Int8(1), 0.0, true, :ok)

function _infeasible_transfer(reason::Symbol)
    return EnergyTransfer(Inf, Int8(0), NaN, false, reason)
end

function _truck_debit(action::TruckDrive, ::EnergyParameters)
    return action.energy
end

function _truck_debit(action::DroneSortie, parameters::EnergyParameters)
    return parameters.shared_energy_factor * action.energy
end

"""Simulate launch-time drone debit, truck driving debit, and full swaps."""
function simulate_energy(
    actions::AbstractVector{<:AbstractEnergyAction},
    initial_energy::Real,
    parameters::EnergyParameters;
    atol::Float64 = 1e-9,
)
    energy = Float64(initial_energy)
    trace = Float64[energy]

    if energy < -atol || energy > parameters.truck_capacity + atol
        return EnergySimulationResult(false, energy, trace, 0, :invalid_initial_energy)
    end

    for (index, action) in enumerate(actions)
        if action isa FullTruckSwap
            energy = parameters.truck_capacity
        elseif action isa DroneSortie
            if action.energy > parameters.drone_capacity + atol
                return EnergySimulationResult(false, energy, trace, index, :drone_capacity)
            end
            debit = _truck_debit(action, parameters)
            if energy + atol < debit
                return EnergySimulationResult(false, energy, trace, index, :shared_battery)
            end
            energy -= debit
        elseif action isa TruckDrive
            debit = _truck_debit(action, parameters)
            if energy + atol < debit
                return EnergySimulationResult(false, energy, trace, index, :truck_battery)
            end
            energy -= debit
        else
            throw(ArgumentError("unsupported energy action $(typeof(action))"))
        end
        abs(energy) <= atol && (energy = 0.0)
        push!(trace, energy)
    end

    return EnergySimulationResult(true, energy, trace, nothing, nothing)
end

function primitive_energy_transfer(
    action::AbstractEnergyAction,
    parameters::EnergyParameters;
    atol::Float64 = 1e-9,
)
    if action isa FullTruckSwap
        return EnergyTransfer(0.0, Int8(0), parameters.truck_capacity, true, :ok)
    elseif action isa DroneSortie
        action.energy <= parameters.drone_capacity + atol ||
            return _infeasible_transfer(:drone_capacity)
        debit = _truck_debit(action, parameters)
    elseif action isa TruckDrive
        debit = _truck_debit(action, parameters)
    else
        throw(ArgumentError("unsupported energy action $(typeof(action))"))
    end

    debit <= parameters.truck_capacity + atol ||
        return _infeasible_transfer(:truck_capacity)
    return EnergyTransfer(debit, Int8(1), -debit, true, :ok)
end

"""Compose two summaries in route order: first `left`, then `right`."""
function compose_energy_transfers(
    left::EnergyTransfer,
    right::EnergyTransfer,
    truck_capacity::Real;
    atol::Float64 = 1e-9,
)
    left.feasible || return left
    right.feasible || return right

    capacity = Float64(truck_capacity)
    if left.slope == 0
        left.intercept + atol >= right.required_input ||
            return _infeasible_transfer(:interface_energy)
        required = left.required_input
    else
        required = max(left.required_input, right.required_input - left.intercept)
    end

    required <= capacity + atol || return _infeasible_transfer(:truck_capacity)
    slope = Int8(right.slope * left.slope)
    intercept = right.slope * left.intercept + right.intercept

    minimum_output = slope * required + intercept
    maximum_output = slope * capacity + intercept
    minimum_output >= -atol || return _infeasible_transfer(:negative_exit_energy)
    maximum_output <= capacity + atol || return _infeasible_transfer(:over_capacity)

    return EnergyTransfer(max(0.0, required), slope, intercept, true, :ok)
end

"""Summarize an action sequence into a constant-size, exactly composable map."""
function summarize_energy(
    actions::AbstractVector{<:AbstractEnergyAction},
    parameters::EnergyParameters,
)
    transfer = identity_energy_transfer()
    for action in actions
        primitive = primitive_energy_transfer(action, parameters)
        transfer = compose_energy_transfers(
            transfer,
            primitive,
            parameters.truck_capacity,
        )
    end
    return transfer
end

"""Return the exact exit energy, or `nothing` when the entry is infeasible."""
function evaluate_energy_transfer(
    transfer::EnergyTransfer,
    initial_energy::Real;
    atol::Float64 = 1e-9,
)
    energy = Float64(initial_energy)
    if !transfer.feasible || energy + atol < transfer.required_input
        return nothing
    end
    output = transfer.slope * energy + transfer.intercept
    return abs(output) <= atol ? 0.0 : output
end
