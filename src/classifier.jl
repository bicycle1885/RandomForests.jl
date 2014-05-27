using MLBase

Tree = Trees.Tree

# internal data structure for a random forest
type RandomForestClassifierLearner
    n_samples::Int
    n_features::Int
    n_max_features::Int
    label_mapping::LabelMap
    improvements::Vector{Float64}
    trees::Vector{Tree}

    function RandomForestClassifierLearner(rf, x, y)
        n_samples, n_features = size(x)

        if n_samples != length(y)
            throw(DimensionMismatch(""))
        end

        n_max_features = begin
            max_features = rf.max_features

            if is(max_features, :sqrt)
                ifloor(sqrt(n_features))
            elseif isa(max_features, Integer)
                max(int(max_features), n_features)
            elseif isa(max_features, FloatingPoint)
                ifloor(n_features * max_features)
            elseif is(max_features, nothing)
                n_features
            else
                error("max_features is invalid: $max_features")
            end
        end

        @assert 0 < n_max_features <= n_features

        label_mapping = labelmap(y)
        improvements = zeros(Float64, n_features)
        trees = Array(Tree, rf.n_estimators)
        new(n_samples, n_features, n_max_features, label_mapping, improvements, trees)
    end
end

type RandomForestClassifier
    # parameters
    n_estimators::Int
    max_features::Any
    max_depth::Int
    min_samples_split::Int

    # learner
    learner::Union(RandomForestClassifierLearner, Nothing)

    function RandomForestClassifier(;n_estimators::Int=100, max_features::Union(Integer, FloatingPoint, Symbol)=:sqrt, max_depth=nothing, min_samples_split::Int=2)
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

        if min_samples_split < 1
            error("min_sample_split is too small (got: $min_samples_split)")
        end

        new(n_estimators, max_features, max_depth, min_samples_split, nothing)
    end
end

function set_weight!(bootstrap::Vector{Int}, sample_weight::Vector{Float64})
    @assert length(bootstrap) == length(sample_weight)
    # initialize weight
    for i in 1:length(sample_weight)
        sample_weight[i] = 0.0
    end
    # set weight
    for i in bootstrap
        sample_weight[i] += 1.0
    end
end

function fit!(rf::RandomForestClassifier, x, y)
    learner = RandomForestClassifierLearner(rf, x, y)
    y_encoded = labelencode(learner.label_mapping, y)
    n_samples = learner.n_samples

    # pre-allocation
    bootstrap = Array(Int, n_samples)
    sample_weight = Array(Float64, n_samples)

    for b in 1:rf.n_estimators
        rand!(1:n_samples, bootstrap)
        set_weight!(bootstrap, sample_weight)
        example = Example(x, y_encoded, sample_weight)
        tree = Trees.Tree()
        Trees.fit!(tree, example, learner.n_max_features, 2)
        learner.trees[b] = tree
    end

    rf.learner = learner
    return
end

function predict(rf::RandomForestClassifier, x)
    if is(rf.learner, nothing)
        error("not yet trained")
    end

    n_samples = size(x, 1)
    output = Array(Int, n_samples)
    n_labels = length(rf.learner.label_mapping)

    for i in 1:n_samples
        counts = zeros(Int, n_labels)
        for b in 1:rf.n_estimators
            tree = rf.learner.trees[b]
            vote = Trees.predict(tree, x[i, :])
            counts[vote] += 1
        end
        output[i] = indmax(counts)
    end

    labeldecode(rf.learner.label_mapping, output)
end
