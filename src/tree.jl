module Trees

using StatsBase
using DataFrames

include("example.jl")
include("sort.jl")

export Tree, fit, predict

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

    function ClassificationLeaf(example::Example, samples::Vector{Int}, impurity::Float64)
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

    function RegressionLeaf(example::Example, samples::Vector{Int}, impurity::Float64)
        new(mean(example.y[samples]), impurity, length(samples))
    end
end

Base.mean(leaf::RegressionLeaf) = leaf.mean

immutable Undef <: Element; end
const undef = Undef()

abstract Criterion
abstract ClassificationCriterion <: Criterion
abstract RegressionCriterion <: Criterion

immutable GiniCriterion <: ClassificationCriterion; end  # Gini index
immutable CrossEntropyCriterion <: ClassificationCriterion; end  # cross entropy
immutable MSECriterion <: RegressionCriterion; end  # Mean Sequared Error
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

# bundled arguments for splitting a node
immutable SplitArgs
    index::Int
    depth::Int
    range::Range{Int}
end

type Splitter
    samples::Vector{Int}
    feature::AbstractVector
    range::Range{Int}
    example::Example
    criterion::Criterion
end

immutable Split{T}
    threshold::T
    boundary::Int
    left_range::Range{Int}
    right_range::Range{Int}
    left_impurity::Float64
    right_impurity::Float64
end

Base.start(sp::Splitter) = 1

function Base.done(sp::Splitter, state)
    # constant feature
    sp.feature[state] == sp.feature[end]
end

function Base.next(sp::Splitter, state)
    n_samples = length(sp.range)
    feature = sp.feature

    # seek for the next boundary
    local threshold, boundary = 0
    for i in state:n_samples-1
        if feature[i] != feature[i+1]
            boundary = i
            threshold = feature[i]
            break
        end
    end

    @assert boundary > 0

    r = sp.range  # shortcut
    left_range = r[1:boundary]
    right_range = r[boundary+1:end]
    left_impurity = impurity(sp.samples[left_range], sp.example, sp.criterion)
    right_impurity = impurity(sp.samples[right_range], sp.example, sp.criterion)
    Split(threshold, boundary, left_range, right_range, left_impurity, right_impurity), boundary + 1
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

function fit(tree::Tree, example::Example, criterion::Criterion, max_features::Int, max_depth::Int, min_samples_split::Int)
    params = Params(criterion, max_features, max_depth, min_samples_split)
    samples = where(example.sample_weight)
    sample_range = 1:length(samples)
    next_index!(tree)
    args = SplitArgs(tree.index, 1, sample_range)
    build_tree(tree, example, samples, args, params)
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

function leaf(example::Example, samples, criterion::RegressionCriterion)
    RegressionLeaf(example, samples, impurity(samples, example, criterion))
end

function leaf(example::Example, samples, criterion::ClassificationCriterion)
    ClassificationLeaf(example, samples, impurity(samples, example, criterion))
end

function build_tree(tree::Tree, example::Example, samples::Vector{Int}, args::SplitArgs, params::Params)
    n_features = example.n_features
    range = args.range  # shortcut
    n_samples = length(range)

    if args.depth >= params.max_depth || n_samples < params.min_samples_split
        tree.nodes[args.index] = leaf(example, samples[range], params.criterion)
        return
    end

    best_feature = 0
    best_impurity = Inf
    local best_threshold, best_boundary

    for k in sample(1:n_features, params.max_features, replace=false)
        feature = example.x[samples[range], k]
        sort!(samples, feature, range)
        splitter = Splitter(samples, feature, range, example, params.criterion)

        for s in splitter
            n_left_samples = length(s.left_range)
            n_right_samples = n_samples - n_left_samples
            averaged_impurity = (s.left_impurity * n_left_samples + s.right_impurity * n_right_samples) / (n_left_samples + n_right_samples)

            if averaged_impurity < best_impurity
                best_impurity = averaged_impurity
                best_feature = k
                best_threshold = s.threshold
                best_boundary = s.boundary
            end
        end
    end

    if best_feature == 0
        tree.nodes[args.index] = leaf(example, samples[range], params.criterion)
    else
        feature = example.x[samples[range], best_feature]
        sort!(samples, feature, range)

        left = next_index!(tree)
        right = next_index!(tree)
        tree.nodes[args.index] = Node(best_feature, best_threshold, best_impurity, n_samples, left, right)

        next_depth = args.depth + 1
        left_node = SplitArgs(left, next_depth, range[1:best_boundary])
        right_node = SplitArgs(right, next_depth, range[best_boundary+1:end])
        build_tree(tree, example, samples, left_node, params)
        build_tree(tree, example, samples, right_node, params)
    end

    return
end

function count_labels(samples::Vector{Int}, example::Example)
    counts = zeros(Float64, example.n_labels)
    n_samples = 0.
    for s in samples
        n_samples += example.sample_weight[s]
        label = example.y[s]
        counts[label] += example.sample_weight[s]
    end

    counts, n_samples
end

function impurity(samples::Vector{Int}, example::Example, ::GiniCriterion)
    counts, n_samples = count_labels(samples, example)
    gini_index = 0.
    for c in counts
        r = c / n_samples
        gini_index += r * r
    end
    1. - gini_index
end

function impurity(samples::Vector{Int}, example::Example, ::CrossEntropyCriterion)
    counts, n_samples = count_labels(samples, example)
    cross_entropy = 0.
    for c in counts
        p = c / n_samples
        if p != 0.
            cross_entropy -= p * log(p)
        end
    end
    cross_entropy
end

function impurity(samples::Vector{Int}, example::Example, ::MSECriterion)
    mean = 0.
    n_samples = 0.

    for s in samples
        mean += example.y[s]
        n_samples += example.sample_weight[s]
    end
    mean /= n_samples

    mse = 0.
    for s in samples
        error = example.y[s] - mean
        mse += error * error
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

end  # module Trees
