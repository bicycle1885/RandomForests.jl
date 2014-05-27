using RDatasets
using RandomForests

iris = dataset("datasets", "iris")
rf = RandomForestClassifier(n_estimators=100, max_features=:sqrt)
fit!(rf, iris[:, 1:4], iris[:, :Species])
@show predict(rf, iris[:, 1:4])
