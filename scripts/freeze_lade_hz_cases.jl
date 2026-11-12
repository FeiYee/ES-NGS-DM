# Freeze Hangzhou demand-grounded cases from CSV coords into .jls geometry.
# Does not invent routes. Depot is origin (ASSUMED). Stations are synthetic.
using Serialization, SHA, DelimitedFiles
const CASEDIR = get(ENV, "ESNGS_LADE_OUT", normpath(joinpath(@__DIR__, "..", "data", "lade_generated")))
const GEOM = joinpath(CASEDIR, "geometry")
mkpath(GEOM)

manifest = joinpath(CASEDIR, "lade_hz_manifest.csv")
isfile(manifest) || error("run build_lade_hz_cases.py first")
lines = readlines(manifest)
header = first(lines)
out_manifest = joinpath(CASEDIR, "lade_hz_geometry_manifest.csv")
open(out_manifest, "w") do io
    println(io, "case_id,kind,n,k,n_parcels,geometry_file,customer_sha256,station_sha256,geometry_sha256,bytes")
    for line in lines[2:end]
        isempty(strip(line)) && continue
        z = split(line, ",")
        id, kind = z[1], z[2]
        n, k = parse(Int, z[3]), parse(Int, z[4])
        n_parcels = parse(Int, z[5])
        cust = readdlm(joinpath(CASEDIR, z[6]), ',', Float64; skipstart=1)
        st = readdlm(joinpath(CASEDIR, z[7]), ',', Float64; skipstart=1)
        size(cust) == (n, 2) || error("customer size mismatch $id")
        size(st) == (k, 2) || error("station size mismatch $id")
        raw = (
            n=n, k=k, seed=0, split="development", demand="HZ", supply="P_synth",
            customers=cust, stations=st, version="lade_hz_demand_grounded_v1",
            case_id=id, kind=kind, n_parcels=n_parcels,
            assumptions="depot_origin;synthetic_peripheral_BSS;all_AOI_drone_eligible",
        )
        path = joinpath("geometry", id * ".jls")
        dest = joinpath(CASEDIR, path)
        open(dest, "w") do f
            serialize(f, raw)
        end
        bytes = read(dest)
        gsha = bytes2hex(sha256(bytes))
        csha = bytes2hex(sha256(reinterpret(UInt8, vec(cust))))
        ssha = bytes2hex(sha256(reinterpret(UInt8, vec(st))))
        println(io, join((id, kind, n, k, n_parcels, replace(path, '\\'=>'/'),
                          csha, ssha, gsha, filesize(dest)), ","))
        println("FROZEN ", id, " n=", n, " k=", k)
    end
end
println("LADE_HZ_GEOMETRY_FROZEN ", out_manifest)
