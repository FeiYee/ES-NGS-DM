# ES-NGS online search: Propose (raw) → Neural Critic top-K → Exact repair.
# Fair online wall = init (charged outside) + this search (includes neural + repair).
using Random, SHA, Dates, Statistics, UUIDs

const ESNGS_ROOT = @__DIR__
const ESNGS_REPRO = normpath(joinpath(ESNGS_ROOT, ".."))
const ESNGS_CKPT = normpath(joinpath(ESNGS_ROOT, "..", "..", "checkpoints", "es_ngs_dm.pt"))
const ESNGS_SCORE_PY = joinpath(ESNGS_ROOT, "score_batch.py")
const ESNGS_SERVER_PY = joinpath(ESNGS_ROOT, "score_server.py")

include(joinpath(ESNGS_REPRO, "sync_block_lns_prototype.jl"))
include(joinpath(ESNGS_ROOT, "features.jl"))
include(joinpath(ESNGS_ROOT, "npy_io.jl"))

mutable struct CriticServer
    proc::Base.Process
    python::String
    ckpt::String
end

function start_critic_server(python::AbstractString, ckpt::AbstractString)
    # Propagate scorer env. ESNGS_FORCE_CPU=1 is for smoke/determinism only.
    # Never invent CUDA_VISIBLE_DEVICES="" in production — that hides the GPU and
    # breaks torch.load of CUDA-saved checkpoints (device_count()==0).
    force_cpu_raw = get(ENV, "ESNGS_FORCE_CPU", "0")
    force_cpu = lowercase(force_cpu_raw) in ("1", "true", "yes")
    cuda_visible = force_cpu ? "" : get(ENV, "CUDA_VISIBLE_DEVICES", nothing)
    withenv(
        "ESNGS_FORCE_CPU" => force_cpu_raw,
        "CUDA_VISIBLE_DEVICES" => cuda_visible,
        "ESNGS_TORCH_THREADS" => get(ENV, "ESNGS_TORCH_THREADS", "1"),
        "PYTHONUNBUFFERED" => "1",
    ) do
        # Accept either an executable path or a launcher specification such as
        # "py -3.10".  Interpolating the raw string would treat the latter as
        # one nonexistent executable on Windows.
        python_text = String(python)
        # A literal Windows path contains backslashes that shell_split would
        # interpret as escapes.  Split only launcher specifications.
        python_argv = isfile(python_text) ? [python_text] : Base.shell_split(python_text)
        isempty(python_argv) && error("empty Python command")
        cmd = Cmd(vcat(python_argv, ["-u", ESNGS_SERVER_PY]))
        proc = open(cmd; read=true, write=true)
        ready = readline(proc)
        startswith(ready, "READY") || error("score_server not ready: $ready")
        println(proc, "LOAD\t$ckpt"); flush(proc)
        # Accept optional LOADING progress lines before OK/ERR.
        resp = ""
        for _ in 1:16
            resp = readline(proc)
            startswith(resp, "OK") && break
            startswith(resp, "ERR") && error("score_server LOAD failed: $resp")
            startswith(resp, "LOADING") && continue
            isempty(resp) && error("score_server LOAD EOF")
            # ignore other progress
        end
        startswith(resp, "OK") || error("score_server LOAD failed: $resp")
        return CriticServer(proc, String(python), String(ckpt))
    end
end

function stop_critic_server!(srv::Union{CriticServer,Nothing})
    isnothing(srv) && return
    try
        println(srv.proc, "QUIT"); flush(srv.proc)
    catch
    end
    try
        # Non-blocking kill — waiting on Python shutdown was inflating search_s past fair wall.
        kill(srv.proc)
    catch
    end
end

function score_via_server!(srv::CriticServer, feats, feat_dir::AbstractString; mask_mode::AbstractString="none")
    # Exclusive per-call directory; always remove after reading scores to avoid
    # multi-worker races on shared X/A/action/global/mask/scores.npy paths.
    mkpath(feat_dir)
    try
        write_npy_c(joinpath(feat_dir, "X.npy"), feats.X)
        write_npy_c(joinpath(feat_dir, "A.npy"), Int64.(feats.A))
        write_npy_c(joinpath(feat_dir, "action.npy"), feats.action)
        write_npy_c(joinpath(feat_dir, "global.npy"), feats.global_feat)
        write_npy_c(joinpath(feat_dir, "mask.npy"), feats.mask)
        out_scores = joinpath(feat_dir, "scores.npy")
        println(srv.proc, "SCORE\t$feat_dir\t$out_scores\t$mask_mode"); flush(srv.proc)
        resp = readline(srv.proc)
        startswith(resp, "OK") || error("SCORE failed: $resp")
        return vec(_read_npy_f64(out_scores))
    finally
        try
            rm(feat_dir; recursive=true, force=true)
        catch
        end
    end
end

function _route_hash(tr, dr)
    bytes2hex(sha256(string(tr, "|", dr)))
end

function _propose_unique(rng, inst, tr, dr; n::Int, window::Int=3)
    out = []; seen = Set{String}(); tries = 0
    while length(out) < n && tries < n * 60
        tries += 1
        p = propose_sync_block_lns(rng, inst, tr, dr; window=window, guided=false)
        isnothing(p) && continue
        h = _route_hash(p[1], p[2])
        h in seen && continue
        push!(seen, h); push!(out, p)
    end
    return out
end

function _read_npy_f64(path::AbstractString)
    open(path, "r") do io
        magic = read(io, 6)
        magic == b"\x93NUMPY" || error("bad npy magic")
        major = read(io, UInt8); minor = read(io, UInt8)
        hlen = major == 1 ? Int(read(io, UInt16)) : Int(read(io, UInt32))
        header = String(read(io, hlen))
        m = match(r"shape': \(([^)]*)\)", header)
        isnothing(m) && error("no shape in $header")
        dims = [parse(Int, strip(x)) for x in split(m.captures[1], ",") if !isempty(strip(x))]
        data = Array{Float64}(undef, dims...)
        # C-order read into Julia column-major via linear fill then reshape note:
        # we write last-dim-fastest; Julia stores first-dim-fastest. Read into flat then permute.
        n = prod(dims)
        flat = Vector{Float64}(undef, n)
        read!(io, flat)
        if length(dims) == 1
            return flat
        elseif length(dims) == 2
            # C: row-major [i,j] with j fastest → Julia A[i,j]
            A = Array{Float64}(undef, dims[1], dims[2])
            idx = 1
            for i in 1:dims[1], j in 1:dims[2]
                A[i, j] = flat[idx]; idx += 1
            end
            return A
        else
            error("unsupported ndim $(length(dims))")
        end
    end
end

"""Cheap geom fallback if critic missing / scoring fails."""
function _cheap_scores(inst, before_tr, props)
    st = Set(inst.station_nodes)
    before_core = [v for v in before_tr if !(v in st)]
    function core_len(core)
        s = 0.0
        for i in 1:(length(core) - 1)
            s += inst.truck_time_matrix[core[i], core[i + 1]]
        end
        return s
    end
    b = core_len(before_core)
    return [100.0 * (b - core_len(p[1])) for p in props]
end

"""guide_mode:
  :full       — neural critic (all features)
  :wo_neural  — geometry proxy ranking (no NN); same propose→top-K→exact
  :random     — random ranking
  :wo_energy  — neural with Energy channels zeroed
  :wo_sync    — neural with Sync channels zeroed
"""
function run_es_ngs_search(inst, initial;
        seconds::Float64=30.0,
        seed::Int=1,
        window::Int=3,
        propose_n::Int=64,
        top_k::Int=1,
        station_path_mode::Symbol=:pareto,
        python::AbstractString="python",
        ckpt::AbstractString=ESNGS_CKPT,
        tmp_root::AbstractString=joinpath(ESNGS_ROOT, "experiments", "tmp_score"),
        guide_mode::Symbol=:full,
        max_rounds::Int=0,
        trace_path::AbstractString="",
        max_exact_evals::Int=0,
        max_empty_rounds::Int=8,
        diversity::Bool=false)
    rng = MersenneTwister(seed)
    graph = station_path_mode == :pareto ? ESNGSDM.build_station_pareto_graph(inst) : nothing
    best = initial
    t0 = time_ns()
    budget = seconds * 1e9
    fixed_iters = max_rounds > 0
    attempts = 0
    generated = 0
    neural_calls = 0
    exact_evals = 0
    improved = 0
    neural_s = 0.0
    repair_s = 0.0
    need_neural = guide_mode in (:full, :wo_energy, :wo_sync) && isfile(ckpt)
    mask_mode = guide_mode == :wo_energy ? "energy" : (guide_mode == :wo_sync ? "sync" : "none")
    round_id = 0
    consecutive_empty_rounds = 0
    total_empty_rounds = 0
    termination_reason = "unknown"
    srv = nothing
    server_startup_s = 0.0
    trace_io = isempty(trace_path) ? nothing : open(trace_path, "w")
    if !isnothing(trace_io)
        println(trace_io, "round,cand_idx,score,selected_rank,prop_hash")
    end
    if need_neural
        try
            server_t0 = time_ns()
            srv = start_critic_server(python, ckpt)
            server_startup_s = (time_ns() - server_t0) / 1e9
            # Model loading is a one-time deployment cost, just like package
            # and JVM startup for the comparison methods.
            t0 += round(Int, server_startup_s * 1e9)
        catch e
            # A neural run must not silently become a geometry-proxy run.
            rethrow(e)
        end
    end

    try
        while true
            if max_exact_evals > 0
                if exact_evals >= max_exact_evals
                    termination_reason = "exact_evaluation_budget"
                    break
                end
            elseif fixed_iters
                if round_id >= max_rounds
                    termination_reason = "round_budget"
                    break
                end
            else
                if !(time_ns() - t0 < budget)
                    termination_reason = "wall_clock_budget"
                    break
                end
                remaining = (budget - (time_ns() - t0)) / 1e9
                if remaining < 0.35
                    termination_reason = "wall_clock_guard"
                    break
                end
            end
            round_id += 1
            n_prop = if fixed_iters
                propose_n
            else
                remaining = (budget - (time_ns() - t0)) / 1e9
                remaining < 2.0 ? max(16, propose_n ÷ 2) : propose_n
            end
            props = _propose_unique(rng, inst, best.truck_route, best.drone_route; n=n_prop, window=window)
            if isempty(props)
                total_empty_rounds += 1
                consecutive_empty_rounds += 1
                if consecutive_empty_rounds >= max_empty_rounds
                    termination_reason = "empty_candidate_stall"
                    break
                end
                continue
            end
            consecutive_empty_rounds = 0
            generated += length(props)
            attempts += 1

            bfrac = fixed_iters ? 1.0 : clamp((budget - (time_ns() - t0)) / 1e9 / seconds, 0.0, 1.0)
            feats = encode_batch(inst, best.truck_route, best.drone_route, best.objective, props, bfrac)

            scores = nothing
            if guide_mode == :random
                scores = rand(rng, length(props))
            elseif guide_mode == :wo_neural
                scores = _cheap_scores(inst, best.truck_route, props)
            elseif need_neural && !isnothing(srv)
                tn = time_ns()
                try
                    # tmp_root must already be exclusive per case/method/seed/worker.
                    # Round subdir + UUID prevents same-process collisions if any.
                    feat_dir = joinpath(tmp_root, "round_$(round_id)_$(guide_mode)_$(uuid4())")
                    scores = score_via_server!(srv, feats, feat_dir; mask_mode=mask_mode)
                    neural_calls += 1
                catch e
                    error("neural scoring failed: $(sprint(showerror, e))")
                end
                neural_s += (time_ns() - tn) / 1e9
            end
            isnothing(scores) && (scores = _cheap_scores(inst, best.truck_route, props))
            length(scores) == length(props) || error("candidate/score count mismatch")
            all(isfinite, scores) || error("nonfinite candidate scores")

            order = sortperm(scores; rev=true)
            k = min(top_k, length(order))
            selected = if diversity && k >= 2
                elite_n = min(6, k)
                chosen = collect(order[1:elite_n])
                pool = collect(order[elite_n+1:min(end, elite_n + 24)])
                while length(chosen) < k && !isempty(pool)
                    best_j = 1
                    best_d = -1.0
                    for (j, idx) in enumerate(pool)
                        dmin = Inf
                        avec = vec(feats.action[idx, :])
                        for sidx in chosen
                            dmin = min(dmin, sum((avec .- vec(feats.action[sidx, :])) .^ 2))
                        end
                        if dmin > best_d
                            best_d = dmin
                            best_j = j
                        end
                    end
                    push!(chosen, pool[best_j])
                    deleteat!(pool, best_j)
                end
                chosen
            else
                collect(order[1:k])
            end
            if !isnothing(trace_io)
                for (rank, idx) in enumerate(order)
                    ph = _route_hash(props[idx][1], props[idx][2])
                    sel = rank <= k ? rank : 0
                    println(trace_io, round_id, ",", idx, ",", scores[idx], ",", sel, ",", ph)
                end
            end
            for idx in selected
                if max_exact_evals > 0 && exact_evals >= max_exact_evals
                    break
                end
                if max_exact_evals == 0 && !fixed_iters
                    rem = (budget - (time_ns() - t0)) / 1e9
                    rem < 0.35 && break
                end
                tr0 = time_ns()
                ev = _eval_composite(inst, props[idx]; station_path_mode=station_path_mode, station_graph=graph)
                repair_s += (time_ns() - tr0) / 1e9
                exact_evals += 1
                isnothing(ev) && continue
                !ev.feasible && continue
                if ev.objective < best.objective - 1e-9
                    best = ev
                    improved += 1
                end
            end
        end
    finally
        stop_critic_server!(srv)
        if !isnothing(trace_io)
            close(trace_io)
        end
    end

    return (
        feasible=best.feasible,
        truck_route=best.truck_route,
        drone_route=best.drone_route,
        attempts=attempts,
        generated=generated,
        exact_evals=exact_evals,
        neural_calls=neural_calls,
        improvements=improved,
        elapsed_seconds=(time_ns() - t0) / 1e9,
        server_startup_s=server_startup_s,
        neural_inference_s=neural_s,
        repair_s=repair_s,
        used_neural=need_neural && neural_calls > 0,
        guide_mode=String(guide_mode),
        empty_rounds=total_empty_rounds,
        termination_reason=termination_reason,
    )
end
