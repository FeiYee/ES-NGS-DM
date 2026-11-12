# Structured candidates: release a block of synchronized truck/drone sorties,
# reinsert drone-only customers into truck, perturb truck visit order, then
# rebuild new drone sorties before energy-DP station repair.
using ESNGSDM,Random,Statistics,Serialization

# Only the sortie energy precheck is needed from the original shared helpers.
# No comparison solver or comparison move library is loaded by this release.
function _valid_sortie_energy(inst, l, c, r)
    e = inst.drone_energy_matrix[l,c] + inst.drone_energy_matrix[c,r]
    return e <= inst.energy_parameters.drone_capacity + 1e-8
end

function _sync_blocks(inst,route,drone)
    d=Set(drone)
    ss=Set(inst.station_nodes)
    core=[x for x in route if !(x in ss)]
    shared=[x for x in core if x in d]
    ti=Dict(x=>i for (i,x) in enumerate(route))
    di=Dict(x=>i for (i,x) in enumerate(drone))
    weights=Float64[]
    for i in 1:(length(shared)-1)
        l,r=shared[i],shared[i+1]
        seg=route[ti[l]:ti[r]]
        ns=Set(inst.station_nodes)
        km=sum(inst.truck_energy_matrix[a,b] for (a,b) in zip(seg,seg[2:end]);init=0.0)
        clean=[v for v in seg if !(v in ns)]
        direct=sum(inst.truck_energy_matrix[a,b] for (a,b) in zip(clean,clean[2:end]);init=0.0)
        detour=max(0.0,km-direct)
        swaps=count(v->v in ns,seg)
        th=km/40+swaps*inst.swap_service_time
        droneh=0.0
        if di[r]-di[l]==2
            cust=drone[di[l]+1]
            droneh=(l==inst.start_depot ? 0.0 : inst.launch_time)+
                inst.drone_time_matrix[l,cust]+
                inst.drone_time_matrix[cust,r]+inst.retrieve_time
        end
        # diagnostic guiding the center only; final objective stays Makespan
        push!(weights,1.0+detour+40*max(th-droneh,0.0)+
                         8*max(droneh-th,0.0))
    end
    return core,shared,weights
end

function _sample_block(rng,weights,window;guided::Bool=true)
    m=length(weights)
    m>=2 || return nothing
    # 15% exploration; rest weighted by observed true station detour and wait.
    center=if !guided || rand(rng)<0.15
        rand(rng,1:m)
    else
        val=rand(rng)*sum(weights)
        found=m
        for i in 1:m
            val-=weights[i]
            if val<=0
                found=i;break
            end
        end
        found
    end
    size=min(window,m)
    start=clamp(center-rand(rng,0:(size-1)),1,m-size+1)
    return start,start+size
end

function _destroy_sync_block(inst,truck,drone,start_shared,end_shared)
    core,shared,_=_sync_blocks(inst,truck,drone)
    l,r=shared[start_shared],shared[end_shared]
    a=findfirst(==(l),drone);b=findfirst(==(r),drone)
    iset=Set(core)
    removed=[v for v in drone[(a+1):(b-1)] if !(v in iset)]
    nc=copy(core)
    # Insert removed drone customers into their original geographic window,
    # at minimal truck-Manhattan insertion increment.
    for c in removed
        from=findfirst(==(l),nc);to=findfirst(==(r),nc)
        best=from;score=Inf
        for i in from:(to-1)
            u,v=nc[i],nc[i+1]
            cost=inst.truck_time_matrix[u,c]+inst.truck_time_matrix[c,v]-
                 inst.truck_time_matrix[u,v]
            if cost<score
                score=cost;best=i
            end
        end
        insert!(nc,best+1,c)
    end
    nd=copy(drone)
    deleteat!(nd,(a+1):(b-1))
    return nc,nd,l,r,length(removed)
end

function _perturb_within_block!(rng,core,l,r)
    for _ in 1:rand(rng,0:3)
        from=findfirst(==(l),core);to=findfirst(==(r),core)
        to>from+3 || return
        i,j=rand(rng,(from+1):(to-1),2)
        if i!=j
            core[i],core[j]=core[j],core[i]
        end
    end
end

function _insert_one_local_sortie(rng,inst,core,drone,l,r;sample_count=30)
    # Reinsert a truck-only customer as a drone customer. Assign new
    # ordered launch/retrieve points, including existing endpoints.
    ll=findfirst(==(l),drone);rr=findfirst(==(r),drone)
    (isnothing(ll)||isnothing(rr)||rr<=ll) && return nothing
    ds=Set(drone)
    slots=Tuple{Int,Int,Int,Int}[]
    for s in ll:(rr-1)
        a,b=drone[s],drone[s+1]
        i=findfirst(==(a),core);j=findfirst(==(b),core)
        (isnothing(i)||isnothing(j)||j<=i+1) && continue
        for k in (i+1):(j-1)
            core[k] in ds && continue
            st=unique([i,max(i,k-2),k-1])
            en=unique([k+1,min(j,k+2),j])
            for p in st,q in en
                (i<=p<k<q<=j) || continue
                x,y,z=core[p],core[k],core[q]
                _valid_sortie_energy(inst,x,y,z) || continue
                push!(slots,(s,k,p,q))
            end
        end
    end
    isempty(slots) && return nothing
    # Prefer avoiding late expensive full DP evaluations by checking
    # transfer geometry first. Maximize proxy truck saved - drone burden.
    best_score=-Inf;chosen=nothing
    for i in rand(rng,1:length(slots),min(length(slots),sample_count))
        s,k,p,q=slots[i]
        a,c,b=core[k-1],core[k],core[k+1]
        saved=inst.truck_time_matrix[a,c]+inst.truck_time_matrix[c,b]-
            inst.truck_time_matrix[a,b]
        dronet=inst.drone_time_matrix[core[p],c]+inst.drone_time_matrix[c,core[q]]
        score=saved-0.30*dronet+0.015*rand(rng)
        if score>best_score
            best_score=score;chosen=slots[i]
        end
    end
    isnothing(chosen) && return nothing
    s,k,p,q=chosen
    oldleft,oldright=drone[s],drone[s+1]
    launch,customer,retrieve=core[p],core[k],core[q]
    new_segment=Int[oldleft]
    launch!=oldleft && push!(new_segment,launch)
    push!(new_segment,customer)
    retrieve!=oldright && push!(new_segment,retrieve)
    push!(new_segment,oldright)
    nc=copy(core);deleteat!(nc,k)
    nd=copy(drone);splice!(nd,s:(s+1),new_segment)
    return nc,nd
end

function propose_sync_block_lns(rng,inst,truck,drone;window=4,
                               max_rebuild=5,guided::Bool=true)
    core,shared,weights=_sync_blocks(inst,truck,drone)
    bl=_sample_block(rng,weights,window;guided=guided)
    isnothing(bl) && return nothing
    i,j=bl
    nc,nd,l,r,removed=_destroy_sync_block(inst,truck,drone,i,j)
    _perturb_within_block!(rng,nc,l,r)
    # Rebuild multiple sorties inside the released contiguous block.
    choices=Int[0,1,2,3,4,5]
    target=min(max_rebuild,max(1,removed+rand(rng,-1:1)))
    rebuilt=0
    for z in 1:target
        cand=_insert_one_local_sortie(rng,inst,nc,nd,l,r)
        isnothing(cand) && break
        nc,nd=cand;rebuilt+=1
    end
    return nc,nd,removed,rebuilt,i,j
end

function _eval_composite(inst,proposal;station_path_mode::Symbol=:bounded,station_graph=nothing)
    isnothing(proposal) && return nothing
    core,drone=proposal[1:2]
    try
        r=repair_fixed_bevtspd_route(inst,core,drone;station_path_mode=station_path_mode,station_graph=station_graph)
        if r.feasible
            chk=evaluate_bevtspd_route(inst,r.truck_route,r.drone_route)
            chk.feasible || error("infeasible but accepted")
            isapprox(r.objective,chk.objective;atol=1e-8) || error("objective mismatch")
            return r
        end
    catch e
        if e isa ArgumentError || e isa BoundsError
            return nothing
        end
        rethrow(e)
    end
    return nothing
end

function run_sync_block_probe(inst,initial;
        seconds::Float64=15.0,seed::Int=20261011,window::Int=4,
        guided::Bool=true,station_path_mode::Symbol=:bounded,
        max_exact_evals::Int=0,max_empty_attempts::Int=1_000_000)
    rng=MersenneTwister(seed)
    graph=station_path_mode==:pareto ? ESNGSDM.build_station_pareto_graph(inst) : nothing
    best=initial; t=time_ns();budget=seconds*1e9
    attempts=0;generated=0;feasible=0;improved=0;exact_evals=0
    consecutive_empty_attempts=0;total_empty_attempts=0
    termination_reason="unknown"
    best_case_info=nothing
    while time_ns()-t<budget
        if max_exact_evals > 0 && exact_evals >= max_exact_evals
            termination_reason="exact_evaluation_budget"
            break
        end
        attempts+=1
        p=propose_sync_block_lns(rng,inst,best.truck_route,best.drone_route;
            window=window,guided=guided)
        if isnothing(p)
            total_empty_attempts+=1
            consecutive_empty_attempts+=1
            if consecutive_empty_attempts>=max_empty_attempts
                termination_reason="empty_candidate_stall"
                break
            end
            continue
        end
        consecutive_empty_attempts=0
        generated+=1
        c=_eval_composite(inst,p;station_path_mode=station_path_mode,station_graph=graph)
        exact_evals += 1
        isnothing(c) && continue
        feasible+=1
        if c.objective<best.objective-1e-9
            best=c;improved+=1;best_case_info=(p[3],p[4],p[5],p[6])
        end
    end
    termination_reason=="unknown" && (termination_reason="wall_clock_budget")
    return (objective=best.objective,feasible=best.feasible,
        truck_route=best.truck_route,drone_route=best.drone_route,
        attempts=attempts,generated=generated,exact_evals=exact_evals,feasible_generated=feasible,
        improvements=improved,elapsed_seconds=(time_ns()-t)/1e9,
        last_win_info=best_case_info,empty_attempts=total_empty_attempts,
        termination_reason=termination_reason)
end

if abspath(PROGRAM_FILE)==@__FILE__
  const ROOT=joinpath(@__DIR__,"benchmark_three_v1")
  const EXTRA=joinpath(@__DIR__,"benchmark_harder_two_v1")
  const CSV=joinpath(@__DIR__,"benchmark_screen_v2","sync_block_lns_prototype_dev.csv")
  open(CSV,"w") do io
    println(io,"n,k,seed,window,feasible,base_h,probe_h,best_gain_pct,attempts,feasible_candidates,improvements,search_seconds,station_visits,station_detour_km")
    for (n,k,folder) in ((100,5,ROOT),(300,8,EXTRA),(500,8,EXTRA))
      seed=20261011
      case=open(deserialize,joinpath(folder,"n$(n)_k$(k)_seed$(seed).jls"))
      inst=case.instance
      initial=solve_energy_icp(inst;chain_initialization_method=:FI)
      initial.feasible || error("bad initial")
      # Pre-determined window sizes, no cherry-picked instances.
      for window in (3,5)
        result=run_sync_block_probe(inst,initial;seconds=12.,seed=seed,window=window)
        check=benchmark_route_metrics(case,result.truck_route,result.drone_route)
        check.feasible || error("bad final route")
        gain=100*(initial.objective-result.objective)/initial.objective
        row=join([n,k,seed,window,true,initial.objective,result.objective,
          gain,result.attempts,result.feasible_generated,result.improvements,
          result.elapsed_seconds,check.station_visits,check.station_detour_km],",")
        println(io,row);flush(io);println("BLOCK_SEARCH ",row)
      end
    end
  end
  println("BLOCK_PROBE_DONE=",CSV)
end
