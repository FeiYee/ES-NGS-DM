root=normpath(joinpath(@__DIR__,".."))
function assert_parsed(x,path)
    if x isa Expr
        x.head in (:error,:incomplete) && error("parse error in $path: $x")
        foreach(y->assert_parsed(y,path),x.args)
    end
end
count=0
for dir in ("src","repro","scripts","tests")
    for (folder,_,files) in walkdir(joinpath(root,dir)), file in files
        endswith(file,".jl") || continue
        path=joinpath(folder,file)
        assert_parsed(Meta.parseall(read(path,String);filename=path),path)
        global count+=1
    end
end
println("JULIA_PARSE_PASS files=",count)
