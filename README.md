# RandomForests.jl

[![Build Status](https://travis-ci.org/bicycle1885/RandomForests.jl.svg?branch=master)](https://travis-ci.org/bicycle1885/RandomForests.jl)

CART-based random forest implementation in Julia.

This package supports:

* Classification model
* Regression model
* Out-of-bag (OOB) error
* Feature importances
* Various configurable parameters

**Please be aware that this package is not yet fully examined implementation. You can use it at your own risk.**
And your bug report or suggestion is welcome!


## Example

Here you can try overview APIs available from the `RandomForests` module.

`example.jl`

```julia
using RDatasets
using RandomForests

# classification
iris = dataset("datasets", "iris")
rf = RandomForestClassifier(n_estimators=100, max_features=:sqrt)
fit(rf, iris[1:4], iris[:Species])
@show predict(rf, iris[1:4])
@show oob_error(rf)
@show feature_importances(rf)


# regression
boston = dataset("MASS", "boston")
rf = RandomForestRegressor(n_estimators=5)
fit(rf, boston[1:13], boston[:MedV])
@show predict(rf, boston[1:13])
```


## Usage

### Models

There are two separate models available in this package - classification and regression.
Each model has its own constructor which is trained by applying the `fit` method.
You can configure these constructors with some keyword arguments listed below:

```julia
RandomForestClassifier(;n_estimators::Int=10,
                        max_features::Union(Integer, FloatingPoint, Symbol)=:sqrt,
                        max_depth=nothing,
                        min_samples_split::Int=2,
                        criterion::Symbol=:gini)
```

```julia
RandomForestRegressor(;n_estimators::Int=10,
                       max_features::Union(Integer, FloatingPoint, Symbol)=:third,
                       max_depth=nothing,
                       min_samples_split::Int=2)
```

* `n_estimators`: the number of weak estimators
* `max_features`: the number of candidate features at each split
    * if `Integer` is given, the fixed number of features are used
    * if `FloatingPoint` is given, the proportion of given value (0.0, 1.0] are used
    * if `Symbol` is given, the number of candidate features is decided by a strategy
        * `:sqrt`: `ifloor(sqrt(n_features))`
        * `:third`: `div(n_features, 3)`
* `max_depth`: the maximum depth of each tree
    * the default argument `nothing` means there is no limitation of the maximum depth
* `min_samples_split`: the minimum number of sub-samples to try to split a node
* `criterion`: the criterion of impurity measure (classification only)
    * `:gini`: Gini index
    * `:entropy`: Cross entropy

`RandomForestRegressor` always uses the mean squared error for its impurity measure.
At the current moment, there is no configurable criteria for regression model.


### Learning / Prediction

Once you create a model, you can easily fit the model using the `fit` method:

```julia
rf = RandomForestClassifier()
fit(rf, x, y)
```

Here the `fit` methods takes three arguments:

* `rf`: the configured model of random forest (`RandomForestClassifier` or `RandomForestRegressor`)
* `x`: the explanatory variables (`AbstractMatrix` or `DataFrame`)
* `y`: the response variable (`AbstractVector`)

Each column of `x` is a feature of the input data and each row is an individual sample.
Each element of `y` is an output corresponding a row of `x`, so the number of row of `x` and the
length of `y` should match.
Note that even though the `DataFrame` object is directly applicable to the `fit` method, applying
a matrix is a much more efficient way to learn quickly.

The prediction using the fitted model is also easy. You can apply the new data to the `predict` method:

```julia
predict(rf, new_x)
```

This returns a vector of predicted values.


### Postmortem

The fitted model includes useful information calculated while learning.

* `oob_error(rf)`: the out-of-bag error, which is known as a good estimator of generalization error
* `feature_importances(rf)`: relative importances of each explanatory variable

The feature importances are normalized values such that the sum of the importances is one.


## Limitations (and ToDo list)

* no parallelism

## Related package

* [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl)
    * DecisionTree.jl is based on the ID3 (Iterative Dichotomiser 3) algorithm while RandomForests.jl uses CART (Classification And Regression Tree).

## Acknowledgement

The algorithm and interface are highly inspired by those of [scikit-learn](http://scikit-learn.org).
