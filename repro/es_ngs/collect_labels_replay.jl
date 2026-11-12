# Replay label expansion: denser search-phase states on the same train instances.
# Writes to ESNGS_LABEL_OUT (default labels_v2_replay). Never touches labels_v2.
using ESNGSDM, Random, Serialization, SHA, Dates, Statistics

const ROOT = @__DIR__
const REPRO = normpath(joinpath(ROOT, ".."))
const OUT = get(ENV, "ESNGS_LABEL_OUT", joinpath(ROOT, "experiments", "labels_v2_replay"))
const CANDS = 64
const WINDOW = 3
# 9 phases ≈ 3× the original early/middle/late triple on the same instances.
const PHASE_SPECS = (
    ("p01", 0.0, 1.0),
    ("p02", 3.0, 0.9),
    ("p03", 6.0, 0.8),
    ("p04", 10.0, 0.7),
    ("p05", 14.0, 0.6),
    ("p06", 18.0, 0.5),
    ("p07", 22.0, 0.4),
    ("p08", 28.0, 0.3),
    ("p09", 34.0, 0.25),
)

include(joinpath(ROOT, "v3_geom_train.jl"))
include(joinpath(REPRO, "sync_block_lns_prototype.jl"))
include(joinpath(ROOT, "features.jl"))
include(joinpath(ROOT, "npy_io.jl"))

say(xs...) = (println(Dates.now(), " ", join(string.(xs), " ")); flush(stdout))
route_hash(tr, dr) = bytes2hex(sha256(string(tr, "|", dr)))
truck_km(inst, tr) = sum(inst.truck_energy_matrix[a, b] for (a, b) in zip(tr, tr[2:end]); init=0.0)
station_visits(inst, tr) = count(v -> v in Set(inst.station_nodes), tr)

function time_components(inst, tr, T)
    drive = truck_km(inst, tr) / 40.0
    swaps = station_visits(inst, tr) * inst.swap_service_time
    return (drive=drive, swap=swaps, sync=T - drive - swaps)
end

function as_sol(r)
    (objective=r.objective, feasible=r.feasible, truck_route=collect(r.truck_route), drone_route=collect(r.drone_route))
end

function advance(inst, sol, seconds, seed)
    seconds <= 0 && return sol
    as_sol(run_sync_block_probe(inst, sol; seconds=Float64(seconds), seed=seed, window=WINDOW, guided=false, station_path_mode=:pareto))
end

function propose_batch(rng, inst, tr, dr; n=CANDS)
    out = []; seen = Set{String}(); tries = 0
    while length(out) < n && tries < n * 50
        tries += 1
        p = propose_sync_block_lns(rng, inst, tr, dr; window=WINDOW, guided=false)
        isnothing(p) && continue
        h = route_hash(p[1], p[2]); h in seen && continue
        push!(seen, h); push!(out, p)
    end
    return out
end

function build_plan()
    plan = []
    seeds200 = 20269901:20269924
    seeds1000 = 20269941:20269956
    fams = [("Q", "P"), ("R", "P"), ("L", "P")]
    for (i, seed) in enumerate(seeds200)
        dem, sup = fams[mod1(i, length(fams))]
        # scale mid/late budgets vs original n200 (8/12) — total advance path ~34s
        push!(plan, (200, 8, seed, dem, sup, 1.0))
    end
    for (i, seed) in enumerate(seeds1000)
        dem, sup = fams[mod1(i, length(fams))]
        push!(plan, (1000, 12, seed, dem, sup, 1.5))
    end
    return plan
end

function run!()
    mkpath(joinpath(OUT, "states"))
    mkpath(joinpath(OUT, "instances"))
    csv_path = joinpath(OUT, "LABELS.csv")
    if isfile(csv_path) && get(ENV, "ESNGS_ALLOW_LABEL_OVERWRITE", "0") != "1"
        error("refusing to overwrite existing labels at $csv_path; set ESNGS_ALLOW_LABEL_OVERWRITE=1 explicitly")
    end
    open(csv_path, "w") do io
        println(io, "state_id,instance_id,n,k,seed,family,phase,candidate_id,before_makespan_h,after_makespan_h,feasible,y_pct,delta_drive_h,delta_swap_h,delta_sync_h,component_resid_h,repair_runtime_s,feat_dir")
    end
    plan = build_plan()
    state_id = 0
    total = 0
    for (n, k, seed, dem, sup, scale) in plan
        fam = dem * sup
        iid = "train_n$(n)_seed$(seed)_$(fam)"
        say("INSTANCE", iid)
        raw = make_train_raw(n, k, seed, dem, sup)
        open(joinpath(OUT, "instances", "$(iid).jls"), "w") do io
            serialize(io, (customers=raw.customers, stations=raw.stations))
        end
        inst = build_bevtspd_instance(raw.customers, raw.stations)
        init = solve_energy_icp(inst; chain_initialization_method=:FI, station_path_mode=:pareto)
        init.feasible || (say("SKIP_INIT", iid); continue)
        sol = as_sol(init)
        graph = ESNGSDM.build_station_pareto_graph(inst)
        prev_t = 0.0
        for (pi, (phase, t_abs, bfrac)) in enumerate(PHASE_SPECS)
            dt = (t_abs - prev_t) * scale
            prev_t = t_abs
            sol = advance(inst, sol, dt, seed + 1000 + pi)
            state_id += 1
            before_T = sol.objective
            before_tr = sol.truck_route
            before_dr = sol.drone_route
            bc = time_components(inst, before_tr, before_T)
            rng = MersenneTwister(seed + state_id * 10007 + 777)
            props = propose_batch(rng, inst, before_tr, before_dr)
            isempty(props) && continue
            feats = encode_batch(inst, before_tr, before_dr, before_T, props, bfrac)
            state_dir = joinpath(OUT, "states", "s$(state_id)")
            mkpath(state_dir)
            open(joinpath(state_dir, "raw_state.jls"), "w") do io
                serialize(io, (
                    instance_id=iid,
                    before_T=before_T,
                    before_tr=before_tr,
                    before_dr=before_dr,
                    props=props,
                    budget_frac=bfrac,
                    replay=true,
                ))
            end
            y = fill(NaN, length(props))
            feas = falses(length(props))
            d_drive = fill(NaN, length(props))
            d_swap = fill(NaN, length(props))
            d_sync = fill(NaN, length(props))
            resid = fill(NaN, length(props))
            open(csv_path, "a") do io
                for (cid, p) in enumerate(props)
                    t0 = time_ns()
                    ev = _eval_composite(inst, p; station_path_mode=:pareto, station_graph=graph)
                    rt = (time_ns() - t0) / 1e9
                    ok = !isnothing(ev) && ev.feasible
                    feas[cid] = ok
                    if ok
                        after_T = ev.objective
                        y[cid] = 100.0 * (before_T - after_T) / before_T
                        ac = time_components(inst, ev.truck_route, after_T)
                        d_drive[cid] = 100.0 * (bc.drive - ac.drive) / before_T
                        d_swap[cid] = 100.0 * (bc.swap - ac.swap) / before_T
                        d_sync[cid] = 100.0 * (bc.sync - ac.sync) / before_T
                        resid[cid] = y[cid] - (d_drive[cid] + d_swap[cid] + d_sync[cid])
                        println(io, join(string.((state_id, iid, n, k, seed, fam, phase, cid, before_T, after_T, true, y[cid], d_drive[cid], d_swap[cid], d_sync[cid], resid[cid], rt, "states/s$(state_id)")), ","))
                    else
                        println(io, join(string.((state_id, iid, n, k, seed, fam, phase, cid, before_T, NaN, false, NaN, NaN, NaN, NaN, NaN, rt, "states/s$(state_id)")), ","))
                    end
                    total += 1
                end
            end
            write_npy_c(joinpath(state_dir, "X.npy"), feats.X)
            write_npy_c(joinpath(state_dir, "A.npy"), Int64.(feats.A))
            write_npy_c(joinpath(state_dir, "action.npy"), feats.action)
            write_npy_c(joinpath(state_dir, "global.npy"), feats.global_feat)
            write_npy_c(joinpath(state_dir, "mask.npy"), feats.mask)
            write_npy_c(joinpath(state_dir, "y.npy"), y)
            write_npy_c(joinpath(state_dir, "feas.npy"), Float64.(feas))
            write_npy_c(joinpath(state_dir, "y_drive.npy"), d_drive)
            write_npy_c(joinpath(state_dir, "y_swap.npy"), d_swap)
            write_npy_c(joinpath(state_dir, "y_sync.npy"), d_sync)
            say("STATE", state_id, phase, "n", n, "cands", length(props), "improve", count(x -> x == x && x > 0, y), "total", total)
        end
    end
    open(joinpath(OUT, "META.txt"), "w") do io
        println(io, "total_labels=", total)
        println(io, "n_states=", state_id)
        println(io, "cands=", CANDS)
        println(io, "feature_version=2")
        println(io, "families=QP,RP,LP")
        println(io, "replay=true")
        println(io, "phases=", join(string.(getindex.(PHASE_SPECS, 1)), ";"))
        println(io, "raw_states_saved=true")
    end
    say("REPLAY_LABELS_DONE", total, "states", state_id)
end

run!()
