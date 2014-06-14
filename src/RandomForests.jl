module RandomForests

using DataFrames

export RandomForestClassifier, RandomForestRegressor, Example, fit!, predict, feature_importances, oob_error

include("example.jl")
include("sort.jl")
include("tree.jl")
include("randomforest.jl")
include("classifier.jl")
include("regressor.jl")

end # RandomForests module
