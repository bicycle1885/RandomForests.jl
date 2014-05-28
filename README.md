# RandomForests.jl

[![Build Status](https://travis-ci.org/bicycle1885/RandomForests.jl.svg?branch=master)](https://travis-ci.org/bicycle1885/RandomForests.jl)

CART-based random forest implementation in Julia.
At the current moment, this is alpha version and runs very ridiculously slow.
But don't be depressed! I'm going to make it run much faster.

## Example

`example.jl`

```julia
using RDatasets
using RandomForests

# classification
iris = dataset("datasets", "iris")
rf = RandomForestClassifier(n_estimators=100, max_features=:sqrt)
fit!(rf, iris[1:4], iris[:Species])
@show predict(rf, iris[1:4])


# regression
boston = dataset("MASS", "boston")
rf = RandomForestRegressor(n_estimators=5)
fit!(rf, boston[1:13], boston[:MedV])
@show predict(rf, boston[1:13])
```

## Limitations (and ToDo list)

* no out-of-bag score
* as for the classifier, Gini index criterion only
* inefficient algorithm
* less configurable parameters (compared to scikit learn)
* parallelism
