type RandomForest{T}
    # parameters
    n_estimators::Int
    max_features::Any
    max_depth::Int
    min_samples_split::Int

    # learner
    learner::Union(T, Nothing)

    function RandomForest(;n_estimators::Int=100, max_features::Union(Integer, FloatingPoint, Symbol)=:sqrt, max_depth=nothing, min_samples_split::Int=2)
        if n_estimators < 1
            error("n_estimators is too small (got: $n_estimators)")
        end

        if isa(max_features, Integer) && max_features < 1
            error("max_features is too small (got: $max_features)")
        elseif isa(max_features, FloatingPoint) && !(0. < max_features <= 1.)
            error("max_features should be in (0, 1] (got: $max_features)")
        elseif isa(max_features, Symbol) && !is(max_features, :sqrt)
            error("max_features should be :sqrt (got: $max_features)")
        end

        if is(max_depth, nothing)
            max_depth = typemax(Int)
        elseif isa(max_depth, Integer) && max_depth <= 1
            error("max_depth is too small (got: $max_depth)")
        end

        if min_samples_split <= 1
            error("min_sample_split is too small (got: $min_samples_split)")
        end

        new(n_estimators, max_features, max_depth, min_samples_split, nothing)
    end
end
