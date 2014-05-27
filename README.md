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

iris = dataset("datasets", "iris")
rf = RandomForestClassifier(n_estimators=100, max_features=:sqrt)
fit!(rf, iris[:, 1:4], iris[:, :Species])
@show predict(rf, iris[:, 1:4])
```

## Limitations (and ToDo list)

* classifier only (no regressor)
* no feature importance
* no out-of-bag score
* Gini index criterion only
* inefficient algorithm
* less configurable parameters (compared to scikit learn)
* parallelism
