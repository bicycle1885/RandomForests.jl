module RandomForests

using DataFrames

export RandomForestClassifier, Example, fit!, predict

include("example.jl")
include("tree.jl")
include("classifier.jl")

end # RandomForests module
