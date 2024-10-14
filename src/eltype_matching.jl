struct EltypeAdaptor{T} end

(l::EltypeAdaptor)(x) = fmap(adapt(l), x)
function (l::EltypeAdaptor)(x::AbstractArray{T}) where {T}
    return isbitstype(T) ? adapt(l, x) : map(l, x)
end

function Adapt.adapt_storage(::EltypeAdaptor{T}, x::AbstractArray) where {T}
    return convert(AbstractArray{T}, x)
end

function Adapt.adapt_storage(::EltypeAdaptor{T}, x::AbstractArray{<:Complex}) where {T}
    return convert(AbstractArray{Complex{T}}, x)
end
