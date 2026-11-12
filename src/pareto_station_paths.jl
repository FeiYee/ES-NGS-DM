# Pareto station multigraph for full truck battery swaps, fixed route legs.
#
# For a path u->a(first station)->...->b(last station)->v, all intermediate
# station-to-station legs start with a full battery. The fastest such path
# for each first/last pair is found by all-pairs shortest path over stations.
# Fixed u->a threshold, fixed post-b energy, and minimal elapsed time
# characterize a globally valid option for any feasible incoming battery.
# None of these station paths are invented: they are actual graph paths.
#
# For full swaps, (required_entry, time, remaining_energy) Pareto dominance
# retains the complete fixed-leg tradeoff frontier. This method is exact for
# arbitrary sequences of station visits within a core arc, unlike bounded
# enumeration in small_exact.jl.

struct StationParetoGraph
    stations::Vector{Int}
    dist::Matrix{Float64}
    next::Matrix{Int}
end

function build_station_pareto_graph(inst::BEVTSPDInstance)
    s=collect(inst.station_nodes)
    k=length(s)
    d=fill(Inf,k,k)
    nxt=zeros(Int,k,k)
    cap=inst.energy_parameters.truck_capacity
    for i in 1:k
        d[i,i]=0.0
        nxt[i,i]=i
        for j in 1:k
            i==j && continue
            if inst.truck_energy_matrix[s[i],s[j]] <= cap+1e-9
                # Arriving at an intermediate station incurs a full swap.
                d[i,j]=inst.truck_time_matrix[s[i],s[j]]+
                       inst.swap_service_time
                nxt[i,j]=j
            end
        end
    end
    # Nonnegative weights: Floyd-Warshall yields the true shortest chain.
    for via in 1:k, i in 1:k, j in 1:k
        trial=d[i,via]+d[via,j]
        if trial<d[i,j]-1e-12
            d[i,j]=trial
            nxt[i,j]=nxt[i,via]
        end
    end
    return StationParetoGraph(s,d,nxt)
end

function _station_pareto_chain(g::StationParetoGraph,i::Int,j::Int)
    isfinite(g.dist[i,j]) || return Int[]
    chain=[g.stations[i]]
    u=i
    for _ in 1:length(g.stations)
        u==j && return chain
        u=g.next[u,j]
        u>0 || error("corrupt station shortest-path predecessor")
        push!(chain,g.stations[u])
    end
    error("station shortest path contains a cycle")
end

function _prune_station_pareto_options(v::Vector{_StationPathOption};
        atol::Float64=1e-10)
    # Direct edges have slope=1, unlike charging paths (slope=0).
    # Cannot compare their final energy by intercept alone.
    by_threshold=sort(v;by=x->(x.time,x.transfer.required_input,
                              -x.transfer.intercept,length(x.nodes)))
    kept=_StationPathOption[]
    for x in by_threshold
        is_swap=x.transfer.slope==0
        dominated=any(y-> y.transfer.slope==x.transfer.slope &&
            y.time<=x.time+atol &&
            y.transfer.required_input<=x.transfer.required_input+atol &&
            y.transfer.intercept>=x.transfer.intercept-atol,kept)
        dominated && continue
        filter!(y-> !(y.transfer.slope==x.transfer.slope &&
            x.time<=y.time+atol &&
            x.transfer.required_input<=y.transfer.required_input+atol &&
            x.transfer.intercept>=y.transfer.intercept-atol),kept)
        push!(kept,x)
    end
    return kept
end

function _build_station_pareto_options(inst::BEVTSPDInstance;
        required_pairs=nothing,
        graph::Union{Nothing,StationParetoGraph}=nothing)
    g=isnothing(graph) ? build_station_pareto_graph(inst) : graph
    nodes=vcat(inst.start_depot,inst.customer_nodes,inst.end_depot)
    pairs=isnothing(required_pairs) ?
        [(a,b) for a in nodes for b in nodes if a!=b] :
        unique(collect(required_pairs))
    cap=inst.energy_parameters.truck_capacity
    ss=g.stations
    out=Dict{Tuple{Int,Int},Vector{_StationPathOption}}()
    # Shortest station-path table is cached in the graph, and the loop
    # enumerates O(|S|^2) first,last station pairs, not permutations.
    chain=Dict{Tuple{Int,Int},Vector{Int}}()
    for i in eachindex(ss),j in eachindex(ss)
        isfinite(g.dist[i,j]) &&
            (chain[(i,j)]=_station_pareto_chain(g,i,j))
    end
    for (from,to) in pairs
        from!=to || throw(ArgumentError("same endpoints"))
        opt=_StationPathOption[]
        direct=inst.truck_energy_matrix[from,to]
        if direct<=cap+1e-9
            push!(opt,_StationPathOption([from,to],
                inst.truck_time_matrix[from,to],
                EnergyTransfer(direct,Int8(1),-direct,true,:ok)))
        end
        for i in eachindex(ss), j in eachindex(ss)
            # Energy required before the FIRST swap.
            reach=inst.truck_energy_matrix[from,ss[i]]
            leave=inst.truck_energy_matrix[ss[j],to]
            (reach<=cap+1e-9 && leave<=cap+1e-9 &&
                 haskey(chain,(i,j))) || continue
            nodes=vcat(from,chain[(i,j)],to)
            time=inst.truck_time_matrix[from,ss[i]]+
                 inst.swap_service_time+g.dist[i,j]+
                 inst.truck_time_matrix[ss[j],to]
            tr=EnergyTransfer(reach,Int8(0),cap-leave,true,:ok)
            push!(opt,_StationPathOption(nodes,time,tr))
        end
        out[(from,to)]=_prune_station_pareto_options(opt)
    end
    return out
end
