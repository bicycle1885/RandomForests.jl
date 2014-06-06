module Trees

using StatsBase

export Tree, fit!, predict

include("example.jl")

abstract Element

type Node{T} <: Element
    feature::Int
    threshold::T
    impurity::Float64
    n_samples::Int
    left::Int
    right::Int
end

abstract Leaf <: Element

type ClassificationLeaf <: Leaf
    counts::Vector{Int}
    impurity::Float64
    n_samples::Int

    function ClassificationLeaf(example, samples::Vector{Int}, impurity::Float64)
        counts = zeros(Int, example.n_labels)
        for s in samples
            label = example.y[s]
            counts[label] += int(example.sample_weight[s])
        end
        new(counts, impurity, length(samples))
    end
end

majority(leaf::ClassificationLeaf) = indmax(leaf.counts)

type RegressionLeaf <: Leaf
    mean::Float64
    impurity::Float64
    n_samples::Int

    function RegressionLeaf(example, samples::Vector{Int}, impurity::Float64)
        new(mean(example.y[samples]), impurity, length(samples))
    end
end

Base.mean(leaf::RegressionLeaf) = leaf.mean

immutable Undef <: Element; end
const undef = Undef()

abstract Criterion

immutable GiniCriterion <: Criterion; end  # Gini index
immutable CrossEntropyCriterion <: Criterion; end  # cross entropy
immutable MSECriterion <: Criterion; end  # Mean Sequared Error
const Gini = GiniCriterion()
const CrossEntropy = CrossEntropyCriterion()
const MSE = MSECriterion()

# parameters to build a tree
immutable Params
    criterion::Criterion
    max_features::Int
    max_depth::Int
    min_samples_split::Int
end

type Tree
    index::Int
    nodes::Vector{Element}

    Tree() = new(0, Element[])
end

getnode(tree::Tree, index::Int) = tree.nodes[index]
getroot(tree::Tree) = getnode(tree, 1)
getleft(tree::Tree, node::Node) = getnode(tree, node.left)
getright(tree::Tree, node::Node) = getnode(tree, node.right)

isnode(tree::Tree, index::Int) = isa(tree.nodes[index], Node)
isleaf(tree::Tree, index::Int) = isa(tree.nodes[index], Leaf)
isnode(node::Element) = isa(node, Node)
isleaf(node::Element) = isa(node, Leaf)

impurity(node::Node) = node.impurity
impurity(leaf::Leaf) = leaf.impurity
n_samples(node::Node) = node.n_samples
n_samples(leaf::Leaf) = leaf.n_samples

function next_index!(tree::Tree)
    push!(tree.nodes, undef)
    tree.index += 1
end

function fit!(tree::Tree, example, criterion::Criterion, max_features::Int, max_depth::Int, min_samples_split::Int)
    params = Params(criterion, max_features, max_depth, min_samples_split)
    samples = where(example.sample_weight)
    next_index!(tree)
    build_tree(tree, example, samples, tree.index, 1, params)
    return
end

function where(v::AbstractVector)
    n = countnz(v)
    indices = Array(Int, n)
    i = 1
    j = 0

    while (j = findnext(v, j + 1)) > 0
        indices[i] = j
        i += 1
    end

    indices
end

function leaf(example, samples, ::GiniCriterion)
    ClassificationLeaf(example, samples, impurity(samples, example, trues(length(example.y)), Gini))
end

function leaf(example, samples, ::CrossEntropyCriterion)
    ClassificationLeaf(example, samples, impurity(samples, example, trues(length(example.y)), CrossEntropy))
end

function leaf(example, samples, ::MSECriterion)
    RegressionLeaf(example, samples, impurity(samples, example, trues(length(example.y)), MSE))
end

function build_tree(tree, example, samples, index, depth, params::Params)
    n_features = example.n_features
    n_samples = length(samples)

    if depth >= params.max_depth || n_samples < params.min_samples_split
        tree.nodes[index] = leaf(example, samples, params.criterion)
        return
    end

    best_feature = 0
    best_impurity = Inf
    local best_threshold, best_split_left::Vector{Int}, best_split_right::Vector{Int}

    for j in sample(1:n_features, params.max_features, replace=false)
        boundaries = unique(example.x[:, j])
        sort!(boundaries)
        pop!(boundaries)
        for boundary in boundaries
            filter = convert(BitVector, example.x[:, j] .<= boundary)
            left_impurity = impurity(samples, example, filter, params.criterion)
            right_impurity = impurity(samples, example, ~filter, params.criterion)
            n_left_samples = countnz(filter[samples])
            n_right_samples = countnz(~filter[samples])
            if n_left_samples == 0 || n_right_samples == 0
                continue
            end
            averaged_impurity = (left_impurity * n_left_samples + right_impurity * n_right_samples) / (n_left_samples + n_right_samples)

            if averaged_impurity < best_impurity
                best_impurity = averaged_impurity
                best_feature = j
                best_threshold = boundary
                best_split_left = intersect(Set(where(filter)), samples)
                best_split_right = [i for i in setdiff(IntSet(samples), IntSet(best_split_left))]
            end
        end
    end

    if best_feature == 0
        tree.nodes[index] = leaf(example, samples, params.criterion)
    else
        left = next_index!(tree)
        right = next_index!(tree)
        tree.nodes[index] = Node(best_feature, best_threshold, best_impurity, n_samples, left, right)
        build_tree(tree, example, best_split_left, left, depth, params)
        build_tree(tree, example, best_split_right, right, depth, params)
    end

    return
end

function count_labels(samples::Vector{Int}, example, filter::BitVector)
    counts = zeros(Float64, example.n_labels)
    n_samples = 0.
    for s in samples
        if filter[s]
            n_samples += example.sample_weight[s]
            label = example.y[s]
            counts[label] += example.sample_weight[s]
        end
    end

    counts, n_samples
end

function impurity(samples::Vector{Int}, example, filter::BitVector, ::GiniCriterion)
    counts, n_samples = count_labels(samples, example, filter)
    gini_index = 0.
    for c in counts
        r = c / n_samples
        gini_index += r * r
    end
    1. - gini_index
end

function impurity(samples::Vector{Int}, example, filter::BitVector, ::CrossEntropyCriterion)
    counts, n_samples = count_labels(samples, example, filter)
    cross_entropy = 0.
    for c in counts
        p = c / n_samples
        if p != 0.
            cross_entropy -= p * log(p)
        end
    end
    #@show cross_entropy
    cross_entropy
end

function impurity(samples::Vector{Int}, example, filter::BitVector, ::MSECriterion)
    mean = 0.
    n_samples = 0.

    for s in samples
        if filter[s]
            mean += example.y[s]
            n_samples += example.sample_weight[s]
        end
    end
    mean /= n_samples

    mse = 0.
    for s in samples
        if filter[s]
            error = example.y[s] - mean
            mse += error * error
        end
    end

    mse / n_samples
end

function predict(tree::Tree, x::AbstractVector)
    node = getroot(tree)

    while true
        if isa(node, Node)
            if x[node.feature] <= node.threshold
                # go left
                node = getleft(tree, node)
            else
                # go right
                node = getright(tree, node)
            end
        elseif isa(node, ClassificationLeaf)
            return majority(node)
        elseif isa(node, RegressionLeaf)
            return mean(node)
        else
            error("found invalid type of node (type: $(typeof(node)))")
        end
    end
end

function predict(tree::Tree, x::DataFrame)
    size(x, 1) == 1 || error("x must be a single record")
    ncol = size(x, 2)
    v = Array(Any, ncol)
    for i in 1:ncol
        v[i] = x[1, i]
    end
    predict(tree, v)
end

end  # module Trees
