using DataFrames

type Example
    x::DataFrame
    y::AbstractVector
    n_labels::Int
    n_features::Int
    sample_weight::Vector{Float64}

    function Example(x, y, sample_weight)
        n_labels = length(unique(y))
        n_features = size(x, 2)
        new(x, y, n_labels, n_features, sample_weight)
    end

    Example(x, y) = Example(x, y, ones(Float64, size(x, 1)))
end
