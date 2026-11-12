# V3 structured spatial-shift benchmark. Generator DOES NOT solve routes.
# All geometry families and severity profiles locked before ANY solver is run.
using Random, Serialization, SHA, Statistics
const ROOT=get(ENV, "ESNGS_GEOMETRY_OUT", normpath(joinpath(@__DIR__, "..", "data", "synthetic_generated")))
const SPECS=((100,5),(200,8),(300,8),(500,8),(1000,12))
const SEEDS=20261011:20261040
const DEMAND=("Q","R","L")  # four corners, ring, paired corridors
const SUPPLY=("C","P")      # centrally biased, peripheral
const LIM=20.0

function demand_coords(n,seed,profile)
  rng=MersenneTwister(seed+n*3001+(profile=="Q" ? 13 : profile=="R" ? 31 : 59))
  x=zeros(n,2)
  for i in 1:n
     if profile=="Q"  # four spatially separated corner clusters
        if rand(rng)<0.10
          x[i,:].=40 .* rand(rng,2) .- 20
        else
          cx=rand(rng,Bool) ? 14.0 : -14.0
          cy=rand(rng,Bool) ? 14.0 : -14.0
          x[i,1]=clamp(cx+2.0*randn(rng),-LIM,LIM)
          x[i,2]=clamp(cy+2.0*randn(rng),-LIM,LIM)
        end
     elseif profile=="R" # distant peripheral annulus
        if rand(rng)<0.10
          x[i,:].=40 .* rand(rng,2) .- 20
        else
          a=2pi*rand(rng)
          rad=14.0+5.0*rand(rng)
          x[i,1]=rad*cos(a); x[i,2]=rad*sin(a)
        end
     elseif profile=="L" # two long east/west delivery corridors
        if rand(rng)<0.10
          x[i,:].=40 .* rand(rng,2) .- 20
        else
          x[i,1]=clamp((rand(rng,Bool) ? 13.0 : -13.0)+2.0*randn(rng),-LIM,LIM)
          x[i,2]=36.0*rand(rng)-18.0
        end
     else
        error("unknown demand profile")
     end
  end
  return x
end

function station_coords(k,n,seed,profile)
  rng=MersenneTwister(seed+n*3007+(profile=="C" ? 71 : 97))
  x=zeros(k,2)
  for j in 1:k
     if rand(rng)<0.25
         x[j,:].=40 .* rand(rng,2) .- 20
     elseif profile=="C"
         # central cluster: no artificial new stations / no change to travel matrix
         x[j,:].=10 .* rand(rng,2) .- 5
     elseif profile=="P"
         ang=2pi*rand(rng);rad=14.0+5.0*rand(rng)
         x[j,1]=rad*cos(ang);x[j,2]=rad*sin(ang)
     else
         error("unknown station profile")
     end
  end
  return x
end
sha_coords(x)=bytes2hex(sha256(reinterpret(UInt8,vec(x))))
mkpath(joinpath(ROOT,"geometry"))
manifest=joinpath(ROOT,"manifest.csv")
isfile(manifest) && error("V3 already frozen; never overwrite.")
nrecords=Ref(0)
open(manifest,"w") do io
    open(joinpath(ROOT,"geometry_metrics.csv"),"w") do stat
      println(io,"case_id,n,k,seed,split,demand,supply,geometry_file,customer_sha256,station_sha256,geometry_sha256,bytes")
      println(stat,"case_id,n,k,demand,supply,mean_nearest_station_km,max_nearest_station_km,customer_edge_fraction")
      for (n,k) in SPECS, seed in SEEDS
        split=seed<=20261020 ? "development" : "holdout"
        for dd in DEMAND, ss in SUPPLY
          customers=demand_coords(n,seed,dd)
          stations=station_coords(k,n,seed,ss)
          @assert size(customers)==(n,2) && size(stations)==(k,2)
          @assert all(z->isfinite(z) && abs(z)<=LIM,customers)
          @assert all(z->isfinite(z) && abs(z)<=LIM,stations)
          id="n$(n)_k$(k)_seed$(seed)_$(dd)$(ss)"
          path=joinpath("geometry",id*".jls")
          raw=(n=n,k=k,seed=seed,split=split,demand=dd,supply=ss,
               customers=customers,stations=stations,version="structured_v3_20261012")
          out=joinpath(ROOT,path)
          open(out,"w") do f
             serialize(f,raw)
          end
          @assert open(deserialize,out).customers==customers
          hash=bytes2hex(sha256(read(out)))
          println(io,join((id,n,k,seed,split,dd,ss,replace(path,'\\'=>'/'),
                       sha_coords(customers),sha_coords(stations),hash,filesize(out)),","))
          near=Float64[]
          for i in 1:n
             push!(near,minimum(hypot(customers[i,1]-stations[j,1],
                     customers[i,2]-stations[j,2]) for j in 1:k))
          end
          edge=count(i->abs(customers[i,1])>12 || abs(customers[i,2])>12,1:n)/n
          println(stat,join((id,n,k,dd,ss,mean(near),maximum(near),edge),","))
          nrecords[]+=1
        end
        flush(io);flush(stat)
      end
    end
end
@assert nrecords[]==900
open(joinpath(ROOT,"SHA256SUMS.txt"),"w") do io
 for file in ("manifest.csv","geometry_metrics.csv")
  println(io,bytes2hex(sha256(read(joinpath(ROOT,file)))),"  ",file)
 end
end
println("STRUCTURED_V3_FROZEN=",nrecords[])
println("DEV=300 HOLDOUT=600; NO OPTIMIZATION ACCESSED")
