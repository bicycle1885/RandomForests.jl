using StatsBase
using MLBase
using Base.Test
using DataFrames
using RandomForests

const testdir = dirname(@__FILE__)

# sample datasets for test
load_iris() = readtable(joinpath(testdir, "data", "iris.csv"))
load_boston() = readtable(joinpath(testdir, "data", "Boston.csv"))

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

let
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

let
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

let
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
    example = RandomForests.Trees.Example{typeof(x)}(x, y)
    tree = RandomForests.Trees.Tree()
    criterion = RandomForests.Trees.Gini
    max_depth = 1000
    min_samples_split = 2
    RandomForests.Trees.fit(tree, example, criterion, n_features, max_depth, min_samples_split)
    for i in 1:n_samples
        @test RandomForests.Trees.predict(tree, vec(x[i, :])) == y[i]
    end
end

let
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
    example = RandomForests.Trees.Example{typeof(x)}(x, y)
    tree = RandomForests.Trees.Tree()
    criterion = RandomForests.Trees.MSE
    max_depth = 1000
    min_samples_split = 2
    RandomForests.Trees.fit(tree, example, criterion, n_features, max_depth, min_samples_split)
    for i in 1:n_samples
        @test_approx_eq RandomForests.Trees.predict(tree, vec(x[i, :])) y[i]
    end
end

let
    srand(1234)

    iris = load_iris()
    samples = 1:150
    variables = 1:4
    output = :Species

    training_samples = sample(samples, 100, replace=false)
    test_samples = filter(i -> i ∉ training_samples, samples)

    for criterion in [:gini, :entropy]
        accs = Float64[]
        errs = Float64[]
        importances = zeros(4)

        for _ in 1:30
            rf = RandomForestClassifier(n_estimators=100, criterion=criterion)
            RandomForests.fit(rf, iris[training_samples, variables], iris[training_samples, output])
            acc = accuracy(iris[test_samples, output], RandomForests.predict(rf, iris[test_samples, variables]))
            push!(accs, acc)
            push!(errs, oob_error(rf))
            importances .+= feature_importances(rf)
        end

        @test mean(accs) > 0.89
        @test abs(mean(errs) - (1 - mean(accs))) < 0.05
        @test sortperm(importances) == [2, 1, 3, 4]
    end
end

let
    srand(1234)

    boston = load_boston()
    n_samples = size(boston, 1)
    samples = 1:n_samples
    variables = 1:13
    output = :MedV

    training_samples = sample(samples, div(n_samples, 2), replace=false)
    test_samples = filter(i -> i ∉ training_samples, samples)

    rmsds = Float64[]
    errs = Float64[]
    for _ in 1:30
        rf = RandomForestRegressor(n_estimators=100)
        RandomForests.fit(rf, boston[training_samples, variables], boston[training_samples, output])
        expected = convert(Vector{Float64}, boston[test_samples, output])
        push!(rmsds, rmsd(RandomForests.predict(rf, boston[test_samples, variables]), expected))
        push!(errs, oob_error(rf))
    end

    @test mean(rmsds) < 5.0
    @test var(rmsds) < 0.05
    @test abs(mean(errs) - mean(rmsds)) < 2.0

    # no test here (is there any good way?)
    #feature_importances(rf)
end

let
    # use an array as input
    srand(0x00)
    iris = load_iris()
    rfc = RandomForestClassifier()
    @test RandomForests.fit(rfc, Array(iris[1:4]), vec(Array(iris[:Species]))) === nothing
    @test isa(RandomForests.predict(rfc, Array(iris[1:4])), Vector{UTF8String})

    boston = load_boston()
    rfr = RandomForestRegressor()
    @test RandomForests.fit(rfr, Array(boston[1:13]), vec(Array(boston[:MedV]))) === nothing
    @test isa(RandomForests.predict(rfr, Array(boston[1:13])), Vector{Float64})
end
