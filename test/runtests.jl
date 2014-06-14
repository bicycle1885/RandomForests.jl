using RDatasets
using StatsBase
using MLBase
using Base.Test
using RandomForests

function accuracy(given::AbstractVector, predicted::AbstractVector)
    @assert length(given) == length(predicted)
    counteq = 0
    for i in 1:endof(given)
        if given[i] == predicted[i]
            counteq += 1
        end
    end
    counteq / length(given)
end

begin
    # default parameters
    rf = RandomForestClassifier()
    @test rf.n_estimators == 10
    @test rf.max_features == :sqrt
    @test rf.max_depth == typemax(Int)
    @test rf.min_samples_split == 2
    @test rf.criterion == RandomForests.Trees.Gini
    @test rf.learner == nothing

    rf = RandomForestRegressor()
    @test rf.n_estimators == 10
    @test rf.max_features == :third
    @test rf.max_depth == typemax(Int)
    @test rf.min_samples_split == 2
    @test rf.criterion == RandomForests.Trees.MSE
    @test rf.learner == nothing
end

begin
    # set parameters
    rf = RandomForestClassifier(n_estimators=10, max_features=.5, max_depth=6, min_samples_split=4, criterion=:entropy)
    @test rf.n_estimators == 10
    @test rf.max_features == .5
    @test rf.max_depth == 6
    @test rf.min_samples_split == 4
    @test rf.criterion == RandomForests.Trees.CrossEntropy

    rf = RandomForestRegressor(n_estimators=20, max_features=10, max_depth=10, min_samples_split=3)
    @test rf.n_estimators == 20
    @test rf.max_features == 10
    @test rf.max_depth == 10
    @test rf.min_samples_split == 3
    @test rf.criterion == RandomForests.Trees.MSE  # default
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
    criterion = RandomForests.Trees.Gini
    max_depth = 1000
    min_samples_split = 2
    RandomForests.Trees.fit(tree, example, criterion, n_features, max_depth, min_samples_split)
    for i in 1:n_samples
        @test RandomForests.Trees.predict(tree, vec(x[i, :])) == y[i]
    end
end

begin
    srand(0x00)
    x = [0 0 0 1;
         0 0 1 1;
         0 1 0 1;
         0 1 1 1;
         1 1 1 2;
         1 1 2 2;
         1 2 1 1;
         1 2 2 2;]
    y = [0., 0., 1., 1., 1., 1., 2., 2.]
    n_samples, n_features = size(x)
    example = Example(x, y)
    tree = RandomForests.Trees.Tree()
    criterion = RandomForests.Trees.MSE
    max_depth = 1000
    min_samples_split = 2
    RandomForests.Trees.fit(tree, example, criterion, n_features, max_depth, min_samples_split)
    for i in 1:n_samples
        @test_approx_eq RandomForests.Trees.predict(tree, vec(x[i, :])) y[i]
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

    # Gini index criterion (default)
    rf = RandomForestClassifier(n_estimators=100)
    fit(rf, iris[training_samples, variables], iris[training_samples, output])
    acc = accuracy(iris[test_samples, output], predict(rf, iris[test_samples, variables]))
    @test acc > .9

    importances = feature_importances(rf)
    @test sortperm(importances) == [2, 1, 3, 4]
    @test importances[2] < .05  # minimum
    @test importances[4] > .35  # maximum

    # the oob error should be close to the test error
    @test abs(oob_error(rf) - (1. - acc)) < .01

    # cross entropy criterion
    srand(0x00)
    rf = RandomForestClassifier(n_estimators=100, criterion=:entropy)
    fit(rf, iris[training_samples, variables], iris[training_samples, output])
    acc = accuracy(iris[test_samples, output], predict(rf, iris[test_samples, variables]))
    @test acc > .9

    importances = feature_importances(rf)
    @test sortperm(importances) == [2, 1, 4, 3]
    @test abs(oob_error(rf) - (1. - acc)) < .01
end

begin
    srand(0x00)

    boston = dataset("MASS", "boston")
    n_samples = size(boston, 1)
    samples = 1:n_samples
    variables = 1:13
    output = :MedV

    training_samples = sample(samples, div(n_samples, 2), replace=false)
    test_samples = filter(i -> i ∉ training_samples, samples)

    rf = RandomForestRegressor(n_estimators=10)
    fit(rf, boston[training_samples, variables], boston[training_samples, output])
    expected = convert(Vector{Float64}, boston[test_samples, output])
    @test rmsd(predict(rf, boston[test_samples, variables]), expected) < 4.

    # no idea
    importances = feature_importances(rf)
    #@show importances
end
