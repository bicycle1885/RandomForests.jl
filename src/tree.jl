module Trees

using StatsBase

export Tree, fit!, predict

include("example.jl")

abstract Element

type Node{T} <: Element
    feature::Int
    threshold::T
    impurity::Float64
    left::Int
    right::Int
end

type Leaf <: Element
    counts::Vector{Int}
    impurity::Float64

    function Leaf(example, samples::Vector{Int}, impurity::Float64)
        counts = zeros(Int, example.n_labels)
        for s in samples
            label = example.y[s]
            counts[label] += int(example.sample_weight[s])
        end
        new(counts, impurity)
    end
end

majority(leaf::Leaf) = indmax(leaf.counts)

immutable Undef <: Element; end
const undef = Undef()

# parameters to build a tree
immutable Params
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

function next_index!(tree::Tree)
    push!(tree.nodes, undef)
    tree.index += 1
end

function fit!(tree::Tree, example, max_features::Int, max_depth::Int, min_samples_split::Int)
    params = Params(max_features, max_depth, min_samples_split)
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

function build_tree(tree, example, samples, index, depth, params::Params)
    n_features = example.n_features
    n_samples = length(samples)

    if depth >= params.max_depth || n_samples < params.min_samples_split
        tree.nodes[index] = Leaf(example, samples, impurity(samples, example, trues(length(example.y))))
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
            left_impurity = impurity(samples, example, filter)
            right_impurity = impurity(samples, example, ~filter)
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
        tree.nodes[index] = Leaf(example, samples, impurity(samples, example, trues(length(example.y))))
    else
        left = next_index!(tree)
        right = next_index!(tree)
        tree.nodes[index] = Node(best_feature, best_threshold, best_impurity, left, right)
        build_tree(tree, example, best_split_left, left, depth, params)
        build_tree(tree, example, best_split_right, right, depth, params)
    end

    return
end

function impurity(samples::Vector{Int}, example, filter::BitVector)
    counts = zeros(Float64, example.n_labels)
    n_samples = 0.
    for s in samples
        if filter[s]
            n_samples += example.sample_weight[s]
            label = example.y[s]
            counts[label] += example.sample_weight[s]
        end
    end

    gini_index = 0.
    for c in counts
        r = c / n_samples
        gini_index += r * r
    end
    1. - gini_index
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
        elseif isa(node, Leaf)
            return majority(node)
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
