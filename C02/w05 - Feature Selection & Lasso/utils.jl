using Random
using MLJ
using DataFrames

"""
Split a given Dataframe into 2 sub-dataframe allowing
for training/testing dataset split
"""
function train_test_split(df::DF; split=0.8, seed=42, shuffled=true) where {DF <: Union{DataFrame, SubDataFrame}}
  Random.seed!(seed)
  (nr, nc) = size(df)
  nrp = round(Int, nr * split)
  row_ixes = shuffled ? shuffle(1:nr) : collect(1:nr)

  df_train = view(df[row_ixes, :], 1:nrp, 1:nc)
  df_test = view(df[row_ixes, :], nrp+1:nr, 1:nc)

  (df_train, df_test)
end

"""
From a Dataframe to matrix
from week2
"""
function get_data(df::DF, features, output) where {DF <: Union{DataFrame, SubDataFrame}}
  df[:, :constant] .= 1.0 # df.constant = fill(1.0, size(df, 1))
  features = [:constant, features...]
  X_matrix = convert(Matrix, select(df, features)) # to get a matrix
  y = df[!, output]                                # => to get a vector
  return (X_matrix, y)
end


"""
Assume feature_matrix is a matrix containing the features as columns
and weights is a corresponding array

from week2
"""
function predict_output(X::Matrix{T}, weights::Vector{T}) where {T <: Real}
    X * weights
end

"""
Calculate RSS given weights vector

depends on predict_output
from week2

y::SubArray{Float64,1, Array{Float64,1}, Tuple{UnitRange{Int64}}, true}
"""
function calc_rss(X::Matrix{T}, y, weights::Vector{T})  where {T <: Real}
  preds = predict_output(X, weights)
  sum((preds .- y) .^ 2)
end


"""
Calculate RSS given a (MLJ) machine which is a wrapper combining a model
and its data.

depends on MLJ predict()

mach::MLJBase.Machine{MLJLinearModels.RidgeRegressor,true}
"""
function get_rss(mach::Machine, X::DF, y::Vector{T}) where {DF <: Union{DataFrame, SubDataFrame}, T <: Real}
  ŷ = predict(mach, X)     # First get the predictions
  diff = y .- ŷ            # Then compute the residuals/errors
  rss = sum(diff .* diff)  # Then square and add them up
  return rss
end
