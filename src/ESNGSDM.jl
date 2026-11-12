# Dependency locations are resolved at load time, not captured in a cache.
__precompile__(false)
module ESNGSDM

using Random, SHA, TOML

include("upstream_backend.jl")
include("energy.jl")
include("energy_tree.jl")
include("energy_route.jl")
include("bevtspd_instance.jl")
include("bevtspd_benchmark.jl")
include("small_exact.jl")
include("pareto_station_paths.jl")
include("energy_icp.jl")
include("energy_tree_search.jl")
include("energy_tree_pair_swap.jl")

export AbstractEnergyAction, TruckDrive, DroneSortie, FullTruckSwap,
    EnergyParameters, EnergySimulationResult, EnergyTransfer, simulate_energy,
    primitive_energy_transfer, compose_energy_transfers, summarize_energy,
    evaluate_energy_transfer, EnergyTransferTree, full_energy_transfer,
    update_energy_action!, query_energy_transfer, evaluate_energy_replacement,
    route_energy_actions, simulate_route_energy, BEVTSPDInstance,
    build_bevtspd_instance, generate_random_bevtspd_instance,
    BEVTSPDBenchmarkCase, paper_reference_station_count, build_paired_bevtspd_case,
    benchmark_route_metrics, evaluate_bevtspd_route, ExactBEVTSPDResult,
    solve_small_bevtspd_exact, EnergyICPResult, EnergyRoleSearchResult,
    repair_fixed_bevtspd_route, solve_energy_icp, improve_drone_to_truck_roles,
    improve_truck_drone_role_exchange, improve_all_truck_order,
    optimize_fixed_truck_order, improve_truck_order_with_sync_dp,
    improve_role_exchange_with_order_search, improve_coupled_energy_neighborhoods,
    TreeScreenedSearchResult, improve_same_ring_adjacent_swaps,
    improve_truck_only_pair_swaps

end
