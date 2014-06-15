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

        n_max_features = resolve_max_features(rf.max_features, n_features)
        @assert 0 < n_max_features <= n_features

        improvements = zeros(Float64, n_features)
        trees = Array(Tree, rf.n_estimators)
        new(n_samples, n_features, n_max_features, improvements, trees)
    end
end

typealias RandomForestRegressor RandomForest{Regressor}

function RandomForestRegressor(;n_estimators::Int=10, max_features::Union(Integer, FloatingPoint, Symbol)=:third, max_depth=nothing, min_samples_split::Int=2)
    RandomForest{Regressor}(n_estimators, max_features, max_depth, min_samples_split, :mse)
end

function fit(rf::RandomForestRegressor, x, y)
    learner = Regressor(rf, x, y)
    n_samples = learner.n_samples

    # pre-allocation
    bootstrap = Array(Int, n_samples)
    sample_weight = Array(Float64, n_samples)

    for b in 1:rf.n_estimators
        rand!(1:n_samples, bootstrap)
        set_weight!(bootstrap, sample_weight)
        example = Trees.Example(x, y, sample_weight)
        tree = Trees.Tree()
        Trees.fit(tree, example, rf.criterion, learner.n_max_features, rf.max_depth, rf.min_samples_split)
        learner.trees[b] = tree
    end

    set_improvements!(learner)
    rf.learner = learner
    return
end

function predict(rf::RandomForestRegressor, x)
    if is(rf.learner, nothing)
        error("not yet trained")
    end

    n_samples = size(x, 1)
    output = Array(Float64, n_samples)
    vs = Array(Float64, rf.n_estimators)

    for i in 1:n_samples
        for b in 1:rf.n_estimators
            tree = rf.learner.trees[b]
            vs[b] = Trees.predict(tree, vec(x[i, :]))
        end
        output[i] = mean(vs)
    end

    output
end
