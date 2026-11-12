# Minimal little-endian C-order .npy writer.
function _npy_header(descr::String, shape::Tuple)
    shape_str = length(shape) == 1 ? "$(shape[1])," : join(shape, ", ")
    header_dict = "{'descr': '$descr', 'fortran_order': False, 'shape': ($shape_str), }"
    pad = 16 - ((10 + length(header_dict) + 1) % 16)
    pad = pad == 16 ? 0 : pad
    return header_dict * " "^pad * "\n"
end

function write_npy_c(path::AbstractString, arr::AbstractArray{Float64})
    mkpath(dirname(path))
    header = _npy_header("<f8", size(arr))
    open(path, "w") do io
        write(io, b"\x93NUMPY", UInt8(1), UInt8(0), UInt16(length(header)), header)
        for idx in CartesianIndices(size(arr))  # first-dim-fastest; we need last-dim-fastest
            nothing
        end
        # explicit last-index-fastest traversal
        _write_c_order!(io, arr)
    end
end

function write_npy_c(path::AbstractString, arr::AbstractArray{Int64})
    mkpath(dirname(path))
    header = _npy_header("<i8", size(arr))
    open(path, "w") do io
        write(io, b"\x93NUMPY", UInt8(1), UInt8(0), UInt16(length(header)), header)
        _write_c_order!(io, arr)
    end
end

function _write_c_order!(io, arr::AbstractArray{T,1}) where {T}
    for i in eachindex(arr)
        write(io, arr[i])
    end
end
function _write_c_order!(io, arr::AbstractArray{T,2}) where {T}
    n1, n2 = size(arr)
    for i1 in 1:n1, i2 in 1:n2
        write(io, arr[i1, i2])
    end
end
function _write_c_order!(io, arr::AbstractArray{T,3}) where {T}
    n1, n2, n3 = size(arr)
    for i1 in 1:n1, i2 in 1:n2, i3 in 1:n3
        write(io, arr[i1, i2, i3])
    end
end
function _write_c_order!(io, arr::AbstractArray{T,4}) where {T}
    n1, n2, n3, n4 = size(arr)
    for i1 in 1:n1, i2 in 1:n2, i3 in 1:n3, i4 in 1:n4
        write(io, arr[i1, i2, i3, i4])
    end
end
