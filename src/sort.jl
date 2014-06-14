import Base: sort!

function sort!(samples::AbstractVector, feature::AbstractVector, range::Range{Int})
    @assert length(feature) == length(range) <= length(samples)
    sort!(sub(samples, range), feature, 1, endof(feature))
end

function sort!(x::AbstractVector, y::AbstractVector, lo::Int, hi::Int)
    if lo <= hi
        p = partition(x, y, lo, hi)
        sort!(x, y, lo, p - 1)
        sort!(x, y, p + 1, hi)
    end
end

function median(x::AbstractVector, i::Int, j::Int, k::Int)
    if x[i] < x[j]
        if x[j] < x[k]
            return j
        elseif x[k] < x[i]
            return i
        else
            return k
        end
    else
        # implies x[j] <= x[i]
        if x[k] <= x[j]
            return j
        elseif x[i] <= x[k]
            return i
        else
            return k
        end
    end
end

function partition(x::AbstractVector, y::AbstractVector, lo::Int, hi::Int)
    # choose pivot
    pivot_index = median(y, lo, hi, div(lo + hi, 2))
    pivot_value = y[pivot_index]

    # swap elements at pivot_index and hi
    y[pivot_index], y[hi] = y[hi], y[pivot_index]
    x[pivot_index], x[hi] = x[hi], x[pivot_index]

    p = lo
    @inbounds for i in lo:hi-1
        if y[i] <= pivot_value
            # swap elements at i and p
            y[i], y[p] = y[p], y[i]
            x[i], x[p] = x[p], x[i]
            p += 1
        end
    end

    # swap elements at p and hi
    y[p], y[hi] = y[hi], y[p]
    x[p], x[hi] = x[hi], x[p]

    p
end
