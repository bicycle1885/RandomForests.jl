type Regressor
    n_samples::Int
    n_features::Int
    n_max_features::Int
    improvements::Vector{Float64}
    trees::Vector{Tree}

    function Regressor(rf, x, y)
        n_samples, n_features = size(x)

        if n_samples != length(y)
            throw(DimensionMismatch(""))
        end

        resolve_max_features(rf.max_features, n_features)
        @assert 0 < n_max_features <= n_features

        improvements = zeros(Float64, n_features)
        trees = Array(Tree, rf.n_estimators)
        new(n_samples, n_features, n_max_features, improvements, trees)
    end
end

typealias RandomForestRegressor RandomForest{Regressor}
