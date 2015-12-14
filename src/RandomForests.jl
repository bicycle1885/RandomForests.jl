module RandomForests

export
    RandomForestClassifier,
    RandomForestRegressor,
    fit,
    predict,
    feature_importances,
    oob_error

using DataFrames

include("util.jl")
include("tree.jl")
include("randomforest.jl")
include("classifier.jl")
include("regressor.jl")

end # RandomForests module
