using RDatasets
using StatsBase
using MLBase
using Base.Test
using RandomForests

function accuracy(given::AbstractVector, predicted::AbstractVector)
    @assert length(given) == length(predicted)
    counteq = 0
    for i in 1:length(given)
        if given[i] == predicted[i]
            counteq += 1
        end
    end
    counteq / length(given)
end

begin
    # default parameters
    rf = RandomForestClassifier()
    @test rf.n_estimators == 100
    @test rf.max_features == :sqrt
    @test rf.max_depth == typemax(Int)
    @test rf.min_samples_split == 2
    @test rf.learner == nothing
end

begin
    # set parameters
    rf = RandomForestClassifier(n_estimators=10, max_features=.5, max_depth=6, min_samples_split=4)
    @test rf.n_estimators == 10
    @test rf.max_features == .5
    @test rf.max_depth == 6
    @test rf.min_samples_split == 4
end

begin
    srand(0x00)
    # test tree
    x = [0 0 0 1;
         0 0 1 1;
         0 1 0 1;
         0 1 1 1;
         1 1 1 2;
         1 1 2 2;
         1 2 1 1;
         1 2 2 2;]
    y = [1, 1, 1, 1, 2, 2, 2, 2]
    n_samples, n_features = size(x)
    example = Example(x, y)
    tree = RandomForests.Trees.Tree()
    max_depth = 1000
    min_samples_split = 2
    RandomForests.Trees.fit!(tree, example, n_features, max_depth, min_samples_split)
    for i in 1:n_samples
        @test RandomForests.Trees.predict(tree, vec(x[i, :])) == y[i]
    end
end

begin
    srand(0x00)

    iris = dataset("datasets", "iris")
    samples = 1:150
    variables = 1:4
    output = :Species

    training_samples = sample(samples, 100, replace=false)
    test_samples = filter(i -> i ∉ training_samples, samples)

    rf = RandomForestClassifier()
    fit!(rf, iris[training_samples, variables], iris[training_samples, output])
    @test accuracy(iris[test_samples, output], predict(rf, iris[test_samples, variables])) > .9
end
