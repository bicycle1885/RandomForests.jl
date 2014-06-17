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
