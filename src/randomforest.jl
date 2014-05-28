import .Trees: getroot, getleft, getright, isnode, n_samples, impurity
type RandomForest{T}
    # parameters
    n_estimators::Int
    max_features::Any
    max_depth::Int
    min_samples_split::Int

    # learner
    learner::Union(T, Nothing)

    function RandomForest(;n_estimators::Int=10, max_features::Union(Integer, FloatingPoint, Symbol)=:sqrt, max_depth=nothing, min_samples_split::Int=2)
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

function resolve_max_features(max_features::Any, n_features::Int)
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

function feature_importances(rf::RandomForest)
    if is(rf.learner, nothing)
        error("not yet trained")
    end
    rf.learner.improvements
end

function set_improvements!(learner)
    improvements = learner.improvements

    for tree in learner.trees
        root = getroot(tree)
        add_improvements!(tree, root, improvements)
    end
    normalize!(improvements)
end

function add_improvements!(tree, node, improvements)
    if isnode(node)
        left = getleft(tree, node)
        right = getright(tree, node)
        n_left_samples = n_samples(left)
        n_right_samples = n_samples(right)
        averaged_impurity = (impurity(left) * n_left_samples + impurity(right) * n_right_samples) / (n_left_samples + n_right_samples)
        improvement = impurity(node) - averaged_impurity
        improvements[node.feature] += improvement

        add_improvements!(tree, left, improvements)
        add_improvements!(tree, right, improvements)
    end
    return
end
