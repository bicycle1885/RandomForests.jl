using MLBase

Tree = Trees.Tree

# internal data structure for a random forest classifier
type Classifier
    n_samples::Int
    n_features::Int
    n_max_features::Int
    label_mapping::LabelMap
    improvements::Vector{Float64}
    trees::Vector{Tree}

    function Classifier(rf, x, y)
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

typealias RandomForestClassifier RandomForest{Classifier}

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
    learner = Classifier(rf, x, y)
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
        Trees.fit!(tree, example, learner.n_max_features, rf.max_depth, rf.min_samples_split)
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
