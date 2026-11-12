# Project-owned integration code. No upstream algorithm implementation is bundled.
const _DEPENDENCY_CONFIG = TOML.parsefile(joinpath(@__DIR__, "..", "configs", "upstream_dependencies.toml"))
const _DEPENDENCY_ROOT = abspath(expanduser(get(ENV, "ESNGS_UPSTREAM_ROOT",
    joinpath(homedir(), ".es_ngs_dm", "upstream"))))

function _verified_external_dependency(key::String)
    spec = _DEPENDENCY_CONFIG["dependencies"][key]
    folder = joinpath(_DEPENDENCY_ROOT, spec["directory"])
    isdir(folder) || error("External solver dependency is missing. Run python scripts/setup_solver_dependencies.py before loading ESNGSDM.")
    project = realpath(joinpath(@__DIR__, ".."))
    external = realpath(folder)
    project_key = Sys.iswindows() ? lowercase(project) : project
    external_key = Sys.iswindows() ? lowercase(external) : external
    separator = Sys.iswindows() ? "\\" : "/"
    (external_key == project_key || startswith(external_key, project_key * separator)) &&
        error("Solver dependencies must be outside the ES-NGS-DM repository: $folder")
    for (name, expected) in spec["files"]
        file = joinpath(folder, split(name, '/')...)
        isfile(file) || error("Missing external solver file: $file")
        bytes2hex(sha256(read(file))) == expected || error("External solver source hash mismatch: $file")
    end
    return external
end

const _ICP_EXTERNAL = _verified_external_dependency("icp")
const _TSPDRONE_EXTERNAL = _verified_external_dependency("tspdrone")

# The public upstream package also has optional binary-solver entry points.
# Load its unchanged FI/partitioning source from the separately installed checkout.
module TSPDrone
const MAX_TIME_LIMIT = Inf
const MAX_DRONE_RANGE = Inf
const _SOURCE = getfield(parentmodule(@__MODULE__), :_TSPDRONE_EXTERNAL)
Base.include(@__MODULE__, joinpath(_SOURCE, "src", "tspd_utils.jl"))
Base.include(@__MODULE__, joinpath(_SOURCE, "src", "tsp_ep_all.jl"))
end

function _external_icp_expression(ex)
    if ex isa Expr && ex.head == :using
        if any(arg -> arg isa Expr && arg.head == :. && length(arg.args) == 1 &&
                arg.args[1] in (:LKH, :Concorde, :TSPDrone), ex.args)
            return nothing
        end
    elseif ex isa Expr && ex.head == :call && ex.args[1] === :include
        name = ex.args[2]
        name isa String || error("Unexpected nonliteral include in pinned upstream source")
        haskey(_DEPENDENCY_CONFIG["dependencies"]["icp"]["files"], "src/" * name) ||
            error("Unverified upstream include: $name")
        return :(_include_external_icp($name))
    end
    return ex
end

function _include_external_icp(name::String)
    Base.include(_external_icp_expression, @__MODULE__, joinpath(_ICP_EXTERNAL, "src", name))
end

_include_external_icp("main.jl")
# Our published initializer explicitly selects FI. Optional binary solvers and
# the upstream NICP model are not exposed as supported deployment configurations.
for methods in (CHAIN_INITIALIZATION_METHODS, CHAINLET_INITIALIZATION_METHODS)
    pop!(methods, :Concorde, nothing)
    pop!(methods, :LKH, nothing)
end
