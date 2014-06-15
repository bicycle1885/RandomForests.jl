module RandomForests

using DataFrames

export RandomForestClassifier, RandomForestRegressor, fit, predict, feature_importances, oob_error

include("util.jl")
include("tree.jl")
include("randomforest.jl")
include("classifier.jl")
include("regressor.jl")

end # RandomForests module
