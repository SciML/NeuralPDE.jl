struct EltypeAdaptor{T} end

(l::EltypeAdaptor)(x) = fmap(adapt(l), x)
function (l::EltypeAdaptor)(x::AbstractArray{T}) where {T}
    return isbitstype(T) ? adapt(l, x) : map(adapt(l), x)
end

function Adapt.adapt_storage(::EltypeAdaptor{T}, x::AbstractArray{<:Number}) where {T}
    return convert(AbstractArray{T}, x)
end

function Adapt.adapt_storage(
        ::EltypeAdaptor{T}, x::AbstractArray{<:Complex{<:Number}}) where {T}
    return convert(AbstractArray{Complex{T}}, x)
end
