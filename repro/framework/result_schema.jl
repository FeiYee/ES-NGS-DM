const SMOKE_CSV_HEADER = join([
    "case_id", "n", "k", "seed", "family", "method", "status",
    "makespan_h", "author_objective", "truck_km", "drone_km", "station_visits", "sorties",
    "coverage_ok", "energy_feasible", "geometry_sha", "solution_sha256",
    "init_s", "preprocess_s", "search_s", "neural_inference_s", "repair_s", "validate_s",
    "total_wall_s", "training_s", "budget_s", "timelimit_hit", "failure_reason", "route_file",
    "code_version", "dependency_note",
], ",")

function append_result_row(csv_path::AbstractString, row::AbstractDict)
    mkpath(dirname(csv_path))
    if !isfile(csv_path)
        open(csv_path, "w") do io
            println(io, SMOKE_CSV_HEADER)
        end
    end
    cols = split(SMOKE_CSV_HEADER, ",")
    open(csv_path, "a") do io
        println(io, join([string(get(row, c, "")) for c in cols], ","))
        flush(io)
    end
end
