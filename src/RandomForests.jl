module RandomForests

using DataFrames

export RandomForestClassifier, Example, fit!, predict

include("example.jl")
include("tree.jl")
include("randomforest.jl")
include("classifier.jl")
include("regressor.jl")

end # RandomForests module
