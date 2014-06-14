function Base.vec(x::DataFrame)
    n, m = size(x)
    n == 1 || error("x must be a single record")
    [x[n, j] for j in 1:m]
end
