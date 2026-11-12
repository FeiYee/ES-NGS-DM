# Portable ES-NGS-DM entry point. Distances are km and times are hours.
using ESNGSDM, Serialization, SHA, Dates, UUIDs
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "repro", "es_ngs", "es_ngs_search.jl"))
include(joinpath(ROOT, "repro", "framework", "validate_bevtspd.jl"))

function options(args)
    d = Dict("benchmark"=>"synthetic", "case"=>"n1000_k12_seed20261014_QP",
        "seed"=>"1", "budget"=>"60", "python"=>"python",
        "ckpt"=>joinpath(ROOT,"checkpoints","es_ngs_dm.pt"),
        "out"=>joinpath(ROOT,"outputs"), "max-rounds"=>"0",
        "max-exact"=>"0", "top-k"=>"1", "propose-n"=>"64")
    iseven(length(args)) || error("expected --key value pairs")
    for i in 1:2:length(args)
        startswith(args[i],"--") || error("expected option: $(args[i])")
        k=args[i][3:end]; haskey(d,k) || error("unknown option --$k")
        d[k]=args[i+1]
    end
    d
end

function run_case(d)
    benchmark=d["benchmark"]
    benchmark in ("synthetic","lade") || error("unknown benchmark")
    base=joinpath(ROOT,"data",benchmark)
    manifest=joinpath(base,benchmark=="synthetic" ? "manifest_all.csv" : "manifest.csv")
    lines=readlines(manifest); names=split(first(lines),",")
    matches=filter(l->startswith(l,d["case"]*","),lines[2:end])
    length(matches)==1 || error("case missing or duplicated in manifest")
    row=Dict(zip(names,split(only(matches),",")))
    path=normpath(joinpath(base,row["geometry_file"]))
    sha=bytes2hex(sha256(read(path)))
    sha==row["geometry_sha256"] || error("geometry SHA256 mismatch")
    raw=open(deserialize,path)
    inst=build_bevtspd_instance(raw.customers,raw.stations;truck_speed=40.,
        drone_speed=60.,truck_capacity=100.,drone_capacity=20.,
        shared_energy_factor=0.4,swap_service_time=2/60)
    isfile(d["ckpt"]) || error("checkpoint not found")
    budget=parse(Float64,d["budget"]); budget>0 || error("budget must be positive")
    seed=parse(Int,d["seed"])
    ti=time_ns()
    initial=solve_energy_icp(inst;chain_initialization_method=:FI,station_path_mode=:pareto)
    init_s=(time_ns()-ti)/1e9
    initial.feasible || error("initial solution is infeasible")
    fixed_rounds=parse(Int,d["max-rounds"]); exact_cap=parse(Int,d["max-exact"])
    budget-init_s-5.0>0 || error("initialization exhausted budget; increase --budget")
    tmp=mktempdir(prefix="esngs_")
    result=nothing
    try
        result=run_es_ngs_search(inst,initial;seconds=budget-init_s-5.0,
            seed=seed+300001,window=3,propose_n=parse(Int,d["propose-n"]),
            top_k=parse(Int,d["top-k"]),python=d["python"],ckpt=abspath(d["ckpt"]),
            tmp_root=tmp,station_path_mode=:pareto,guide_mode=:full,
            max_rounds=fixed_rounds,max_exact_evals=exact_cap,diversity=false)
    finally
        rm(tmp;recursive=true,force=true)
    end
    v=validate_bevtspd(inst,result.truck_route,result.drone_route;
        geometry_sha=sha,expected_geometry_sha=sha)
    v.feasible || error("independent feasibility validation failed")
    result.used_neural || error("neural scorer was not used")
    result.neural_calls==result.attempts || error("incomplete neural scoring")
    charged=init_s+result.elapsed_seconds+v.validate_s
    fixed=(fixed_rounds>0 || exact_cap>0)
    !fixed && charged>budget+0.5 && error("charged time exceeded budget: $charged")
    mkpath(d["out"])
    stem=d["case"]*"_s$(seed)_"*string(uuid4())
    serialize(joinpath(d["out"],stem*".jls"),
        (truck_route=result.truck_route,drone_route=result.drone_route,
        completion_h=v.makespan_h,geometry_sha=sha))
    csv=joinpath(d["out"],stem*".csv")
    open(csv,"w") do io
        println(io,"case_id,benchmark,seed,completion_h,truck_km,charged_online_s,init_s,search_s,neural_s,repair_s,validation_s,model_startup_s,exact_evals,generated,attempts,improvements,feasible,budget_s,protocol")
        println(io,join((d["case"],benchmark,seed,v.makespan_h,v.truck_km,charged,init_s,
            result.elapsed_seconds,result.neural_inference_s,result.repair_s,v.validate_s,
            result.server_startup_s,result.exact_evals,result.generated,result.attempts,
            result.improvements,v.feasible,budget,fixed ? "fixed_work" : "charged_wall"),","))
    end
    println("RESULT=",csv)
    println("COMPLETION_H=",v.makespan_h," EXACT_REPAIRS=",result.exact_evals," FEASIBLE=",v.feasible)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_case(options(ARGS))
end
