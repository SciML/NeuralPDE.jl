struct EltypeAdaptor{T} end

function ensure_same_device(x, device)
    if (typeof(x) != device) && !(x isa Number)
        error("Device mismatch detected. Ensure all data is on the same device.")
    end
    return x
end


(l::EltypeAdaptor)(x) = fmap(y -> ensure_same_device(y, l), x)

function (l::EltypeAdaptor)(x::AbstractArray{T}) where {T}
    return (isbitstype(T) || T <: Number) ? x : map(y -> ensure_same_device(y, l), x)
end

function Adapt.adapt_storage(::EltypeAdaptor{T}, x::AbstractArray) where {T}
    return convert(AbstractArray{T}, x)
end

function Adapt.adapt_storage(::EltypeAdaptor{T}, x::AbstractArray{Union{}}) where {T}
    return convert(AbstractArray{T}, x)
end

function Adapt.adapt_storage(::EltypeAdaptor{T}, x::AbstractArray{<:Complex}) where {T}
    return convert(AbstractArray{Complex{T}}, x)
end
