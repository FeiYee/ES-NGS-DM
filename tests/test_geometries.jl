using Serialization,SHA
root=normpath(joinpath(@__DIR__,".."))
count=0
for (name,manifest) in (("synthetic","manifest_all.csv"),("lade","manifest.csv"))
    base=joinpath(root,"data",name)
    lines=readlines(joinpath(base,manifest)); fields=split(first(lines),",")
    for line in lines[2:end]
        row=Dict(zip(fields,split(line,",")))
        path=joinpath(base,row["geometry_file"])
        bytes2hex(sha256(read(path)))==row["geometry_sha256"] || error("bad hash: $path")
        raw=open(deserialize,path)
        size(raw.customers)==(parse(Int,row["n"]),2) || error("bad customer count")
        size(raw.stations)==(parse(Int,row["k"]),2) || error("bad station count")
        all(isfinite,raw.customers) && all(isfinite,raw.stations) || error("nonfinite geometry")
        global count+=1
    end
end
println("FROZEN_GEOMETRY_PASS cases=",count)
